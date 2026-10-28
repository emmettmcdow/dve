//! Where dve's log lines go. Every line dve writes passes through here.
//!
//! With no sink set, a line goes to `std.log` under the `.dve` scope, so a Zig program
//! controls it through its `std_options` like any other library's logging. With a sink set,
//! the line is formatted and handed to the sink instead; the C bindings' `dve_set_log_callback`
//! is a thin wrapper over `setLogFn`.
//!
//! Either way, which levels are compiled in at all is the program's `std_options.log_level`.
//! The sink's `min_level` filters at runtime, below that.

const std = @import("std");

/// Receives one formatted line, without a trailing newline. `msg` is only valid for the
/// duration of the call.
///
/// Called on whichever thread logged, including the engine's work queue, and sometimes while
/// the engine holds its locks: it must be thread-safe and must not call back into dve.
pub const LogFn = *const fn (level: std.log.Level, msg: [:0]const u8, ctx: ?*anyopaque) void;

const Sink = struct {
    f: LogFn,
    ctx: ?*anyopaque,
    min_level: std.log.Level,
};

var sink: ?Sink = null;

/// Sends dve's log lines to `f`, dropping those less severe than `min_level`. Null puts back
/// the default, `std.log`.
///
/// Process-wide, and not synchronized against logging: set it before creating an engine, or
/// while none is running.
pub fn setLogFn(f: ?LogFn, ctx: ?*anyopaque, min_level: std.log.Level) void {
    sink = if (f) |fp| .{ .f = fp, .ctx = ctx, .min_level = min_level } else null;
}

/// Longest line a sink receives, terminator included. Longer ones are cut short and end in
/// `TRUNCATED`.
pub const MAX_LINE = 1024;
const TRUNCATED = "...";

pub fn err(comptime format: []const u8, args: anytype) void {
    write(.err, format, args);
}

pub fn warn(comptime format: []const u8, args: anytype) void {
    write(.warn, format, args);
}

pub fn info(comptime format: []const u8, args: anytype) void {
    write(.info, format, args);
}

pub fn debug(comptime format: []const u8, args: anytype) void {
    write(.debug, format, args);
}

const std_log = std.log.scoped(.dve);

fn write(comptime level: std.log.Level, comptime format: []const u8, args: anytype) void {
    if (!comptime std.log.logEnabled(level, .dve)) return;
    const s = sink orelse return switch (level) {
        .err => std_log.err(format, args),
        .warn => std_log.warn(format, args),
        .info => std_log.info(format, args),
        .debug => std_log.debug(format, args),
    };
    if (@backingInt(level) > @backingInt(s.min_level)) return;

    var buf: [MAX_LINE]u8 = undefined;
    s.f(level, formatLine(&buf, format, args), s.ctx);
}

fn formatLine(buf: *[MAX_LINE]u8, comptime format: []const u8, args: anytype) [:0]const u8 {
    // One byte held back for the terminator.
    var w: std.Io.Writer = .fixed(buf[0 .. MAX_LINE - 1]);
    var len: usize = undefined;
    if (w.print(format, args)) |_| {
        len = w.buffered().len;
    } else |_| {
        len = MAX_LINE - 1;
        @memcpy(buf[len - TRUNCATED.len .. len], TRUNCATED);
    }
    buf[len] = 0;
    return buf[0..len :0];
}

const testing = std.testing;

const Captured = struct {
    level: ?std.log.Level = null,
    msg: std.ArrayList(u8) = .empty,
    calls: usize = 0,

    fn record(level: std.log.Level, msg: [:0]const u8, ctx: ?*anyopaque) void {
        const self: *Captured = @ptrCast(@alignCast(ctx.?));
        self.level = level;
        self.calls += 1;
        self.msg.clearRetainingCapacity();
        self.msg.appendSlice(testing.allocator, msg) catch @panic("OOM");
        // The terminator is part of the contract the C bindings rely on.
        std.debug.assert(msg.ptr[msg.len] == 0);
    }
};

test "a sink receives formatted lines at its level and above" {
    var cap: Captured = .{};
    defer cap.msg.deinit(testing.allocator);
    setLogFn(Captured.record, &cap, .warn);
    defer setLogFn(null, null, .debug);

    err("opened {d} files", .{3});
    try testing.expectEqual(.err, cap.level.?);
    try testing.expectEqualStrings("opened 3 files", cap.msg.items);

    warn("w", .{});
    try testing.expectEqual(.warn, cap.level.?);

    info("dropped", .{});
    debug("dropped", .{});
    try testing.expectEqual(2, cap.calls);
}

test "a line longer than MAX_LINE is cut short and marked" {
    var cap: Captured = .{};
    defer cap.msg.deinit(testing.allocator);
    setLogFn(Captured.record, &cap, .debug);
    defer setLogFn(null, null, .debug);

    const long: [MAX_LINE * 2]u8 = @splat('a');
    err("{s}", .{&long});
    try testing.expectEqual(MAX_LINE - 1, cap.msg.items.len);
    try testing.expect(std.mem.endsWith(u8, cap.msg.items, TRUNCATED));
}

test "a line exactly MAX_LINE - 1 long is not marked" {
    var cap: Captured = .{};
    defer cap.msg.deinit(testing.allocator);
    setLogFn(Captured.record, &cap, .debug);
    defer setLogFn(null, null, .debug);

    const fits: [MAX_LINE - 1]u8 = @splat('a');
    err("{s}", .{&fits});
    try testing.expectEqualStrings(&fits, cap.msg.items);
}

test "clearing the sink stops delivery" {
    var cap: Captured = .{};
    defer cap.msg.deinit(testing.allocator);
    setLogFn(Captured.record, &cap, .debug);
    setLogFn(null, null, .debug);

    // Goes to std.log, which the test runner only prints below its own level.
    debug("to std.log", .{});
    try testing.expectEqual(0, cap.calls);
}
