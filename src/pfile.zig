//! Positional file IO, straight onto libc.
//!
//! Nothing here is clever. It exists because the POSIX file syscalls are a stable, forty-year
//! -old interface and the Zig standard library's file abstraction is not: the buffered
//! Reader/Writer layer was rewritten wholesale in 0.15. This module depends on `open`,
//! `close`, `pread`, `pwrite`, `lseek`, `ftruncate`, `fsync` and `fcntl` and nothing else.
//!
//! Three things the standard library was doing for us that we now have to do ourselves, and
//! that are easy to get wrong:
//!
//!   1. A `pread`/`pwrite` may transfer fewer bytes than asked for, even on a regular file.
//!      Every call site loops.
//!   2. Either may fail with EINTR when a signal lands mid-call. Every call site retries.
//!   3. On Darwin, `fsync` only pushes data to the drive -- it does not make the drive flush
//!      its own write cache, so a power loss can still lose an fsync'd write. `F_FULLFSYNC`
//!      is the one that actually commits, and it is what a database wants.

const builtin = @import("builtin");
const std = @import("std");

const is_darwin = builtin.os.tag.isDarwin();
const is_linux = builtin.os.tag == .linux;

comptime {
    if (!is_darwin and !is_linux) {
        @compileError("pfile supports Darwin and Linux; add the constants for " ++
            @tagName(builtin.os.tag));
    }
    // We hand 64-bit offsets to pread/pwrite/ftruncate directly. On a 32-bit Linux libc
    // those take a 32-bit off_t unless _FILE_OFFSET_BITS=64, which we cannot set from here.
    if (@sizeOf(usize) != 8) @compileError("pfile assumes a 64-bit off_t");
}

// ****************************************************************************** libc bindings
// `open` and `fcntl` are variadic in C. Declaring them with fixed parameters happens to work
// on x86-64, but not on AArch64 macOS, where variadic arguments are passed on the stack while
// fixed ones are passed in registers -- so the callee would read the mode from the wrong
// place. They have to be declared variadic and called with the extra argument.
const c = struct {
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
    extern "c" fn openat(dirfd: c_int, path: [*:0]const u8, flags: c_int, ...) c_int;
    extern "c" fn close(fd: c_int) c_int;
    extern "c" fn pread(fd: c_int, buf: [*]u8, nbyte: usize, offset: i64) isize;
    extern "c" fn pwrite(fd: c_int, buf: [*]const u8, nbyte: usize, offset: i64) isize;
    extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
    extern "c" fn ftruncate(fd: c_int, length: i64) c_int;
    extern "c" fn fsync(fd: c_int) c_int;
    extern "c" fn fcntl(fd: c_int, cmd: c_int, ...) c_int;
};

/// errno is thread-local and reached through a function, not a global, so that threads do not
/// clobber each other's. The accessor's name is the libc's business and differs by platform.
const errnoLocation = if (is_darwin)
    struct {
        extern "c" fn __error() *c_int;
    }.__error
else
    struct {
        extern "c" fn __errno_location() *c_int;
    }.__errno_location;

fn errno() c_int {
    return errnoLocation().*;
}

const O = struct {
    const RDWR: c_int = 0x0002;
    const CREAT: c_int = if (is_darwin) 0x0200 else 0o100;
    const TRUNC: c_int = if (is_darwin) 0x0400 else 0o1000;
    const EXCL: c_int = if (is_darwin) 0x0800 else 0o200;
    /// Without this the descriptor survives an exec, leaking the database into any child
    /// process the host application happens to spawn.
    const CLOEXEC: c_int = if (is_darwin) 0x01000000 else 0o2000000;
};

const SEEK_END: c_int = 2;
const F_FULLFSYNC: c_int = 51; // Darwin only

/// The low errno numbers are shared BSD heritage and agree across both platforms; the ones
/// above ~34 do not, so those are split.
const E = struct {
    const PERM: c_int = 1;
    const NOENT: c_int = 2;
    const INTR: c_int = 4;
    const IO: c_int = 5;
    const BADF: c_int = 9;
    const ACCES: c_int = 13;
    const EXIST: c_int = 17;
    const NOTDIR: c_int = 20;
    const ISDIR: c_int = 21;
    const INVAL: c_int = 22;
    const NFILE: c_int = 23;
    const MFILE: c_int = 24;
    const FBIG: c_int = 27;
    const NOSPC: c_int = 28;
    const ROFS: c_int = 30;

    const NAMETOOLONG: c_int = if (is_darwin) 63 else 36;
    const LOOP: c_int = if (is_darwin) 62 else 40;
    const DQUOT: c_int = if (is_darwin) 69 else 122;
    const OVERFLOW: c_int = if (is_darwin) 84 else 75;
};

// ************************************************************************************* Errors
pub const OpenError = error{
    FileNotFound,
    AccessDenied,
    PathAlreadyExists,
    IsDir,
    NotDir,
    NameTooLong,
    SymLinkLoop,
    ProcessFdQuotaExceeded,
    SystemFdQuotaExceeded,
    ReadOnlyFileSystem,
    NoSpaceLeft,
    InputOutput,
    Unexpected,
};

pub const ReadError = error{
    InputOutput,
    IsDir,
    BadFileDescriptor,
    Unexpected,
};

pub const WriteError = error{
    NoSpaceLeft,
    DiskQuota,
    FileTooBig,
    InputOutput,
    BadFileDescriptor,
    Unexpected,
};

pub const SeekError = error{
    Overflow,
    BadFileDescriptor,
    Unexpected,
};

pub const SyncError = error{
    InputOutput,
    BadFileDescriptor,
    Unexpected,
};

fn unexpected(what: []const u8, e: c_int) error{Unexpected} {
    std.log.warn("pfile: {s} failed with errno {d}", .{ what, e });
    return error.Unexpected;
}

// *************************************************************************************** File
pub const File = struct {
    fd: c_int,

    pub const Opts = struct {
        /// Create the file if it is not there. The mode is masked by the process umask, as
        /// always.
        create: bool = true,
        /// Truncate an existing file to zero length.
        truncate: bool = false,
        /// Fail if the file already exists.
        exclusive: bool = false,
        mode: c_uint = 0o644,
    };

    /// Opens for reading and writing. Paths are relative to the process working directory
    /// unless `dir_fd` is supplied, in which case they are relative to that directory.
    pub fn openAt(dir_fd: ?c_int, path: []const u8, opts: Opts) OpenError!File {
        var buf: [4096]u8 = undefined;
        if (path.len >= buf.len) return error.NameTooLong;
        @memcpy(buf[0..path.len], path);
        buf[path.len] = 0;
        const path_z: [*:0]const u8 = @ptrCast(&buf);

        var flags: c_int = O.RDWR | O.CLOEXEC;
        if (opts.create) flags |= O.CREAT;
        if (opts.truncate) flags |= O.TRUNC;
        if (opts.exclusive) flags |= O.EXCL;

        while (true) {
            const fd = if (dir_fd) |d|
                c.openat(d, path_z, flags, opts.mode)
            else
                c.open(path_z, flags, opts.mode);
            if (fd >= 0) return .{ .fd = fd };
            return switch (errno()) {
                E.INTR => continue,
                E.NOENT => error.FileNotFound,
                E.ACCES, E.PERM => error.AccessDenied,
                E.EXIST => error.PathAlreadyExists,
                E.ISDIR => error.IsDir,
                E.NOTDIR => error.NotDir,
                E.NAMETOOLONG => error.NameTooLong,
                E.LOOP => error.SymLinkLoop,
                E.MFILE => error.ProcessFdQuotaExceeded,
                E.NFILE => error.SystemFdQuotaExceeded,
                E.ROFS => error.ReadOnlyFileSystem,
                E.NOSPC => error.NoSpaceLeft,
                E.IO => error.InputOutput,
                else => |e| unexpected("open", e),
            };
        }
    }

    pub fn close(self: File) void {
        // Retrying close on EINTR is a portability trap: on Linux the descriptor is already
        // gone, so a retry would close whatever fd got that number next. Report and move on.
        if (c.close(self.fd) != 0) {
            std.log.warn("pfile: close failed with errno {d}", .{errno()});
        }
    }

    /// Reads into `buf` starting at `offset`. Returns the number of bytes read, which is
    /// short only at end of file -- a partial transfer for any other reason is retried here
    /// rather than handed to the caller.
    pub fn readAt(self: File, buf: []u8, offset: u64) ReadError!usize {
        var done: usize = 0;
        while (done < buf.len) {
            const n = c.pread(self.fd, buf.ptr + done, buf.len - done, @intCast(offset + done));
            if (n == 0) break; // end of file
            if (n > 0) {
                done += @intCast(n);
                continue;
            }
            switch (errno()) {
                E.INTR => continue,
                E.IO => return error.InputOutput,
                E.ISDIR => return error.IsDir,
                E.BADF => return error.BadFileDescriptor,
                else => |e| return unexpected("pread", e),
            }
        }
        return done;
    }

    /// Writes all of `bytes` at `offset`. Writing past the end of the file extends it, and
    /// POSIX guarantees the gap reads back as zeros -- which is what lets a store rely on a
    /// fresh slot being zero-filled.
    pub fn writeAt(self: File, bytes: []const u8, offset: u64) WriteError!void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.pwrite(self.fd, bytes.ptr + done, bytes.len - done, @intCast(offset + done));
            if (n > 0) {
                done += @intCast(n);
                continue;
            }
            // errno is only meaningful after a -1 return. A zero-byte write with bytes still
            // outstanding is not something a regular file should do, and reading errno here
            // would be reading whatever the last failed call left behind -- if that happened
            // to be EINTR we would spin forever.
            if (n == 0) return unexpected("pwrite", 0);
            switch (errno()) {
                E.INTR => continue,
                E.NOSPC => return error.NoSpaceLeft,
                E.DQUOT => return error.DiskQuota,
                E.FBIG => return error.FileTooBig,
                E.IO => return error.InputOutput,
                E.BADF => return error.BadFileDescriptor,
                else => |e| return unexpected("pwrite", e),
            }
        }
    }

    pub fn size(self: File) SeekError!u64 {
        // lseek moves the descriptor's shared offset, which is harmless here: every read and
        // write this module performs is positional and ignores that offset entirely.
        const n = c.lseek(self.fd, 0, SEEK_END);
        if (n >= 0) return @intCast(n);
        return switch (errno()) {
            E.OVERFLOW => error.Overflow,
            E.BADF => error.BadFileDescriptor,
            else => |e| unexpected("lseek", e),
        };
    }

    /// Extends or shrinks the file. Extending zero-fills.
    pub fn setSize(self: File, length: u64) WriteError!void {
        while (true) {
            if (c.ftruncate(self.fd, @intCast(length)) == 0) return;
            switch (errno()) {
                E.INTR => continue,
                E.NOSPC => return error.NoSpaceLeft,
                E.DQUOT => return error.DiskQuota,
                E.FBIG, E.INVAL => return error.FileTooBig,
                E.IO => return error.InputOutput,
                E.BADF => return error.BadFileDescriptor,
                else => |e| return unexpected("ftruncate", e),
            }
        }
    }

    /// Commits written bytes to stable storage.
    ///
    /// On Darwin `fsync` hands the data to the drive and returns without waiting for the
    /// drive to empty its own write cache, so a power failure can still lose it. F_FULLFSYNC
    /// is the request that actually waits. It is not supported on every filesystem -- network
    /// mounts in particular -- so a failure falls back to plain fsync rather than giving up.
    pub fn sync(self: File) SyncError!void {
        return syncFd(self.fd);
    }
};

/// `File.sync` for a descriptor this module does not own, so a caller holding a `std.fs.File`
/// or a `std.fs.Dir` can get the same durability without handing over the fd's lifetime.
///
/// Directories are the reason this is public. A rename is only as durable as the directory
/// that records it: the renamed file's own `sync` says nothing about whether the new name
/// survived, so a crash can leave the old name pointing at the old contents. Syncing the
/// containing directory afterwards is what commits the swap.
pub fn syncFd(fd: c_int) SyncError!void {
    if (is_darwin) {
        if (c.fcntl(fd, F_FULLFSYNC, @as(c_int, 0)) != -1) return;
    }
    while (true) {
        if (c.fsync(fd) == 0) return;
        switch (errno()) {
            E.INTR => continue,
            E.IO => return error.InputOutput,
            E.BADF => return error.BadFileDescriptor,
            else => |e| return unexpected("fsync", e),
        }
    }
}

// ****************************************************************************************** Tests
const tmpDir = std.testing.tmpDir;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

fn openTmp(dir: std.fs.Dir, name: []const u8, opts: File.Opts) !File {
    return File.openAt(@intCast(dir.fd), name, opts);
}

test "create, write, read back" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();

    const f = try openTmp(tmpD.dir, "a.bin", .{});
    defer f.close();

    try f.writeAt("hello world", 0);
    var buf: [11]u8 = undefined;
    try expectEqual(@as(usize, 11), try f.readAt(&buf, 0));
    try std.testing.expectEqualStrings("hello world", &buf);
    try expectEqual(@as(u64, 11), try f.size());
}

test "reads are positional and do not disturb each other" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    const f = try openTmp(tmpD.dir, "a.bin", .{});
    defer f.close();

    try f.writeAt("0123456789", 0);
    var buf: [3]u8 = undefined;
    _ = try f.readAt(&buf, 7);
    try std.testing.expectEqualStrings("789", &buf);
    _ = try f.readAt(&buf, 0);
    try std.testing.expectEqualStrings("012", &buf);
    // size() uses lseek, which moves the shared descriptor offset. Positional reads must not
    // care.
    _ = try f.size();
    _ = try f.readAt(&buf, 2);
    try std.testing.expectEqualStrings("234", &buf);
}

test "a read past the end is short, not an error" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    const f = try openTmp(tmpD.dir, "a.bin", .{});
    defer f.close();

    try f.writeAt("abc", 0);
    var buf: [16]u8 = undefined;
    try expectEqual(@as(usize, 3), try f.readAt(&buf, 0));
    try expectEqual(@as(usize, 0), try f.readAt(&buf, 3));
    try expectEqual(@as(usize, 0), try f.readAt(&buf, 9999));
}

test "writing past the end zero-fills the gap" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    const f = try openTmp(tmpD.dir, "a.bin", .{});
    defer f.close();

    try f.writeAt("x", 4096);
    try expectEqual(@as(u64, 4097), try f.size());

    var buf: [4097]u8 = undefined;
    @memset(&buf, 0xAA);
    try expectEqual(@as(usize, 4097), try f.readAt(&buf, 0));
    for (buf[0..4096]) |b| try expectEqual(@as(u8, 0), b);
    try expectEqual(@as(u8, 'x'), buf[4096]);
}

test "setSize extends with zeros and shrinks" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    const f = try openTmp(tmpD.dir, "a.bin", .{});
    defer f.close();

    try f.writeAt("abcdef", 0);
    try f.setSize(8192);
    try expectEqual(@as(u64, 8192), try f.size());

    var buf: [16]u8 = undefined;
    _ = try f.readAt(&buf, 0);
    try std.testing.expectEqualStrings("abcdef", buf[0..6]);
    for (buf[6..16]) |b| try expectEqual(@as(u8, 0), b);

    try f.setSize(3);
    try expectEqual(@as(u64, 3), try f.size());
    try expectEqual(@as(usize, 3), try f.readAt(&buf, 0));
}

test "sync commits without error" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    const f = try openTmp(tmpD.dir, "a.bin", .{});
    defer f.close();
    try f.writeAt("durable", 0);
    try f.sync();
}

test "contents survive close and reopen" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    {
        const f = try openTmp(tmpD.dir, "a.bin", .{});
        defer f.close();
        try f.writeAt("persisted", 0);
        try f.sync();
    }
    const f = try openTmp(tmpD.dir, "a.bin", .{});
    defer f.close();
    var buf: [9]u8 = undefined;
    _ = try f.readAt(&buf, 0);
    try std.testing.expectEqualStrings("persisted", &buf);
}

test "opening without create reports FileNotFound" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    try std.testing.expectError(
        error.FileNotFound,
        openTmp(tmpD.dir, "nope.bin", .{ .create = false }),
    );
}

test "an exclusive open of an existing file fails" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    (try openTmp(tmpD.dir, "a.bin", .{})).close();
    try std.testing.expectError(
        error.PathAlreadyExists,
        openTmp(tmpD.dir, "a.bin", .{ .exclusive = true }),
    );
}

test "truncate opens an existing file at zero length" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    {
        const f = try openTmp(tmpD.dir, "a.bin", .{});
        defer f.close();
        try f.writeAt("stale", 0);
    }
    const f = try openTmp(tmpD.dir, "a.bin", .{ .truncate = true });
    defer f.close();
    try expectEqual(@as(u64, 0), try f.size());
}

test "an over-long path is rejected before it reaches libc" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    const long = "x" ** 5000;
    try std.testing.expectError(error.NameTooLong, openTmp(tmpD.dir, long, .{}));
}

test "a large transfer completes in full" {
    var tmpD = tmpDir(.{});
    defer tmpD.cleanup();
    const f = try openTmp(tmpD.dir, "big.bin", .{});
    defer f.close();

    const n = 4 * 1024 * 1024;
    const out = try std.testing.allocator.alloc(u8, n);
    defer std.testing.allocator.free(out);
    for (out, 0..) |*b, i| b.* = @truncate(i);
    try f.writeAt(out, 0);

    const in = try std.testing.allocator.alloc(u8, n);
    defer std.testing.allocator.free(in);
    try expectEqual(n, try f.readAt(in, 0));
    try expect(std.mem.eql(u8, out, in));
}
