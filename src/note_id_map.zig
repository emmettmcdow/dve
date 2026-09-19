pub const NoteID = u64;
// Suffixed by storage type alongside the vector database it belongs to. The note ids in
// here index that database specifically, so sharing one manifest between an unquantized
// and a quantized build would let `pruneOrphanedPaths` drop paths that only the other
// build still has vectors for.
pub const MANIFEST_FILENAME = ".dve_ids" ++ db_suffix;
const MANIFEST_VERSION: u32 = 1;

pub const Error = error{
    NotFound,
    CorruptManifest,
};

/// How the workspace root moved, for `reroot`.
///
/// Paths in the manifest are relative to the workspace root, so moving the root re-expresses
/// every one of them. Which of the two this is depends only on where the new root sits relative
/// to the old one, and the caller is the one holding both absolute paths.
pub const Reroot = union(enum) {
    /// The root rose, and the notes are now this much further down it. Moving a workspace from
    /// `~/web/md-src` up to `~/web` makes every note deeper by `md-src`.
    deeper: []const u8,
    /// The root descended into this subfolder. Notes outside it are no longer in the workspace
    /// at all, and `reroot` reports them so their vectors can go too.
    shallower: []const u8,
};

pub const NoteIdMap = struct {
    path_to_id: std.StringHashMap(NoteID),
    id_to_path: std.AutoHashMap(NoteID, []u8),
    next_id: NoteID,
    allocator: std.mem.Allocator,
    basedir: std.fs.Dir,
    mutex: std.Thread.Mutex,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, basedir: std.fs.Dir) !Self {
        var self = Self{
            .path_to_id = std.StringHashMap(NoteID).init(allocator),
            .id_to_path = std.AutoHashMap(NoteID, []u8).init(allocator),
            .next_id = 1,
            .allocator = allocator,
            .basedir = basedir,
            .mutex = .{},
        };

        self.load() catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        var it = self.id_to_path.valueIterator();
        while (it.next()) |path_ptr| {
            self.allocator.free(path_ptr.*);
        }
        self.path_to_id.deinit();
        self.id_to_path.deinit();
    }

    pub fn getOrCreateId(self: *Self, path: []const u8) !NoteID {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.path_to_id.get(path)) |id| {
            return id;
        }

        const id = self.next_id;
        self.next_id += 1;

        const path_copy = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(path_copy);

        try self.path_to_id.put(path_copy, id);
        errdefer _ = self.path_to_id.remove(path_copy);

        try self.id_to_path.put(id, path_copy);
        errdefer _ = self.id_to_path.remove(id);

        try self.save();

        return id;
    }

    pub fn getId(self: *Self, path: []const u8) ?NoteID {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.path_to_id.get(path);
    }

    pub fn getPath(self: *Self, id: NoteID) ?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.id_to_path.get(id);
    }

    pub fn removePath(self: *Self, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.removePathLocked(path);
    }

    fn removePathLocked(self: *Self, path: []const u8) !void {
        const id = self.path_to_id.get(path) orelse return;

        const owned_path = self.id_to_path.get(id) orelse return;
        _ = self.path_to_id.remove(owned_path);
        _ = self.id_to_path.remove(id);
        self.allocator.free(owned_path);

        self.save() catch |e| {
            std.log.err(
                "Failed to save mapping after removing path '{s}', with error: {}\n",
                .{ path, e },
            );
            return e;
        };
    }

    pub fn renamePath(self: *Self, old_path: []const u8, new_path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const id = self.path_to_id.get(old_path) orelse return error.NotFound;

        _ = self.path_to_id.remove(old_path);

        if (self.id_to_path.getPtr(id)) |path_ptr| {
            self.allocator.free(path_ptr.*);
            path_ptr.* = try self.allocator.dupe(u8, new_path);
            try self.path_to_id.put(path_ptr.*, id);
        }

        try self.save();
    }

    /// Re-express every path against a workspace root that has moved, and save once.
    ///
    /// This is what makes moving a workspace cheap. A note's vectors are keyed by its id, and
    /// the path appears nowhere but here, so a root that moves costs one rewrite of this
    /// manifest rather than re-reading and re-embedding every note behind it.
    ///
    /// Ids the move puts outside the workspace are appended to `dropped` and removed from the
    /// map; their vectors are the caller's to delete, this layer having no reach into them.
    pub fn reroot(self: *Self, move: Reroot, dropped: *std.ArrayList(NoteID)) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Every new path is built before a single old one is touched. A failure halfway through
        // an in-place rewrite would leave some entries relative to the old root and some to the
        // new, and nothing afterwards could tell which was which.
        const Pending = struct { id: NoteID, path: []u8 };
        var pending: std.ArrayList(Pending) = .{};
        defer pending.deinit(self.allocator);
        errdefer for (pending.items) |item| self.allocator.free(item.path);

        const drop_from = dropped.items.len;
        errdefer dropped.shrinkRetainingCapacity(drop_from);

        var it = self.id_to_path.iterator();
        while (it.next()) |entry| {
            const id = entry.key_ptr.*;
            const old_path = entry.value_ptr.*;
            const new_path: []u8 = switch (move) {
                .deeper => |prefix| try std.fs.path.join(self.allocator, &.{ prefix, old_path }),
                .shallower => |prefix| blk: {
                    const rest = stripDir(old_path, prefix) orelse {
                        try dropped.append(self.allocator, id);
                        continue;
                    };
                    break :blk try self.allocator.dupe(u8, rest);
                },
            };
            errdefer self.allocator.free(new_path);
            try pending.append(self.allocator, .{ .id = id, .path = new_path });
        }

        // Room for the rebuilt index before anything is dismantled, so the commit below cannot
        // fail partway.
        try self.path_to_id.ensureTotalCapacity(@intCast(pending.items.len));

        for (dropped.items[drop_from..]) |id| {
            if (self.id_to_path.fetchRemove(id)) |kv| self.allocator.free(kv.value);
        }
        // `path_to_id`'s keys are the very buffers `id_to_path` owns, so it is rebuilt wholesale
        // rather than edited around the paths being freed underneath it.
        self.path_to_id.clearRetainingCapacity();
        for (pending.items) |item| {
            const slot = self.id_to_path.getPtr(item.id).?;
            self.allocator.free(slot.*);
            slot.* = item.path;
            self.path_to_id.putAssumeCapacity(item.path, item.id);
        }
        // Ownership has moved into the map; the errdefer above must not free them now.
        pending.clearRetainingCapacity();

        try self.save();
    }

    pub fn count(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.path_to_id.count();
    }

    /// `path` with a leading `dir/` removed, or null when it is not under `dir` at all.
    fn stripDir(path: []const u8, dir: []const u8) ?[]const u8 {
        if (dir.len == 0) return path;
        if (!std.mem.startsWith(u8, path, dir)) return null;
        const rest = path[dir.len..];
        // A prefix has to end at a separator: `md-src` must not claim `md-srcery/a.md`.
        if (rest.len < 2 or rest[0] != '/') return null;
        return rest[1..];
    }

    // Possibly update the save and load to use SOA. We want to write the fields of the struct all
    // at once, we should be able to call one syscall to write all of the data.
    fn load(self: *Self) !void {
        const zone = tracy.beginZone(@src(), .{ .name = "note_id_map.zig:load" });
        defer zone.end();

        const file = try self.basedir.openFile(MANIFEST_FILENAME, .{});
        defer file.close();

        var rbuf: [4096]u8 = undefined;
        var reader = file.reader(&rbuf);

        const version = try readIntLE(&reader, u32);
        if (version != MANIFEST_VERSION) return error.CorruptManifest;

        self.next_id = try readIntLE(&reader, u64);
        const entry_count = try readIntLE(&reader, u64);

        for (0..entry_count) |_| {
            const id = try readIntLE(&reader, u64);
            const path_len = try readIntLE(&reader, u32);

            const path = try self.allocator.alloc(u8, path_len);
            errdefer self.allocator.free(path);

            try reader.interface.readSliceAll(path);

            try self.path_to_id.put(path, id);
            try self.id_to_path.put(id, path);
        }
    }

    fn save(self: *Self) !void {
        const zone = tracy.beginZone(@src(), .{ .name = "note_id_map.zig:save" });
        defer zone.end();

        const tmp_name = MANIFEST_FILENAME ++ ".tmp";

        const file = try self.basedir.createFile(tmp_name, .{});
        errdefer self.basedir.deleteFile(tmp_name) catch {}; // zlinter-disable-current-line

        var wbuf: [4096]u8 = undefined;
        var writer = file.writer(&wbuf);

        try writeIntLE(&writer, u32, MANIFEST_VERSION);
        try writeIntLE(&writer, u64, self.next_id);
        try writeIntLE(&writer, u64, self.id_to_path.count());

        var it = self.id_to_path.iterator();
        while (it.next()) |entry| {
            try writeIntLE(&writer, u64, entry.key_ptr.*);
            try writeIntLE(&writer, u32, @intCast(entry.value_ptr.len));
            try writer.interface.writeAll(entry.value_ptr.*);
        }

        try writer.interface.flush();
        try file.sync();
        file.close();

        try self.basedir.rename(tmp_name, MANIFEST_FILENAME);
    }

    pub fn pruneOrphanedPaths(self: *Self, basedir: std.fs.Dir) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var to_remove: std.ArrayList(NoteID) = .{};
        defer to_remove.deinit(self.allocator);

        var it = self.path_to_id.iterator();
        while (it.next()) |entry| {
            basedir.access(entry.key_ptr.*, .{}) catch |err| switch (err) {
                error.FileNotFound => {
                    try to_remove.append(self.allocator, entry.value_ptr.*);
                },
                else => {},
            };
        }

        for (to_remove.items) |id| {
            if (self.id_to_path.get(id)) |path| {
                try self.removePathLocked(path);
            }
        }
    }
};

const testing_allocator = std.testing.allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualStrings = std.testing.expectEqualStrings;

test "getOrCreateId creates new id" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    const id1 = try map.getOrCreateId("note1.md");
    const id2 = try map.getOrCreateId("note2.md");

    try expect(id1 != id2);
    try expectEqual(id1, map.getId("note1.md").?);
    try expectEqual(id2, map.getId("note2.md").?);
}

test "getOrCreateId returns existing id" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    const id1 = try map.getOrCreateId("note1.md");
    const id2 = try map.getOrCreateId("note1.md");

    try expectEqual(id1, id2);
}

test "getPath returns path for id" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    const id = try map.getOrCreateId("mypath.md");
    const path = map.getPath(id);

    try expectEqualStrings("mypath.md", path.?);
}

test "removePath removes mapping" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    const id = try map.getOrCreateId("note.md");
    try map.removePath("note.md");

    try expect(map.getId("note.md") == null);
    try expect(map.getPath(id) == null);
}

test "persistence across restarts" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    const id1: NoteID = blk: {
        var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
        defer map.deinit();
        break :blk try map.getOrCreateId("persistent.md");
    };

    var map2 = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map2.deinit();

    try expectEqual(id1, map2.getId("persistent.md").?);
    try expectEqualStrings("persistent.md", map2.getPath(id1).?);

    const id2 = try map2.getOrCreateId("new.md");
    try expect(id2 > id1);
}

test "renamePath preserves id" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    const id = try map.getOrCreateId("old.md");
    try map.renamePath("old.md", "new.md");

    try expect(map.getId("old.md") == null);
    try expectEqual(id, map.getId("new.md").?);
    try expectEqualStrings("new.md", map.getPath(id).?);
}

test "getOrCreateId does not double free when save fails" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    // Remove write permission on the tmpdir so save() cannot create the .tmp file.
    // path_copy is inserted into id_to_path before save() is called; the errdefer
    // then frees it, leaving a dangling value. deinit() frees it again — double free.
    // std.testing.allocator (GPA) will catch the second free.
    try std.posix.fchmod(tmpD.dir.fd, 0o555);
    defer std.posix.fchmod(tmpD.dir.fd, 0o755) catch {};

    const result = map.getOrCreateId("test.md");
    try std.testing.expectError(error.AccessDenied, result);
}

test "thread safety: concurrent access across full interface" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    // Pre-populate; create real files for every other entry so pruneOrphanedPaths has work to do.
    const pre_count = 20;
    for (0..pre_count) |i| {
        var buf: [32]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "pre{d}.md", .{i});
        _ = try map.getOrCreateId(path);
        if (i % 2 == 0) (try tmpD.dir.createFile(path, .{})).close();
    }

    // getOrCreateId — 4 threads each insert 10 unique paths; collect IDs to assert uniqueness.
    const creator_count = 4;
    const creator_ops = 10;
    var creator_ids: [creator_count * creator_ops]NoteID = undefined;

    const CreatorCtx = struct { map: *NoteIdMap, ids: []NoteID, thread_idx: usize, ops: usize };
    const creator_worker = struct {
        fn run(ctx: CreatorCtx) void {
            for (0..ctx.ops) |i| {
                var buf: [32]u8 = undefined;
                const path = std.fmt.bufPrint(&buf, "c{d}_{d}.md", .{ ctx.thread_idx, i }) catch unreachable;
                ctx.ids[ctx.thread_idx * ctx.ops + i] = ctx.map.getOrCreateId(path) catch 0;
            }
        }
    }.run;

    // getId, getPath, count — read concurrently while writers mutate.
    const ReaderCtx = struct { map: *NoteIdMap };
    const reader_worker = struct {
        fn run(ctx: ReaderCtx) void {
            for (0..pre_count) |i| {
                var buf: [32]u8 = undefined;
                const path = std.fmt.bufPrint(&buf, "pre{d}.md", .{i}) catch unreachable;
                if (ctx.map.getId(path)) |id| _ = ctx.map.getPath(id);
                _ = ctx.map.count();
            }
        }
    }.run;

    // removePath — delete the first half of the pre-populated entries.
    const RemoverCtx = struct { map: *NoteIdMap };
    const remover_worker = struct {
        fn run(ctx: RemoverCtx) void {
            for (0..pre_count / 2) |i| {
                var buf: [32]u8 = undefined;
                const path = std.fmt.bufPrint(&buf, "pre{d}.md", .{i}) catch unreachable;
                ctx.map.removePath(path) catch {};
            }
        }
    }.run;

    // renamePath — rename the second half of the pre-populated entries.
    const RenamerCtx = struct { map: *NoteIdMap };
    const renamer_worker = struct {
        fn run(ctx: RenamerCtx) void {
            for (pre_count / 2..pre_count) |i| {
                var old_buf: [32]u8 = undefined;
                var new_buf: [32]u8 = undefined;
                const old_path = std.fmt.bufPrint(&old_buf, "pre{d}.md", .{i}) catch unreachable;
                const new_path = std.fmt.bufPrint(&new_buf, "ren{d}.md", .{i}) catch unreachable;
                ctx.map.renamePath(old_path, new_path) catch {};
            }
        }
    }.run;

    // pruneOrphanedPaths — scans and removes entries whose files are absent.
    const PrunerCtx = struct { map: *NoteIdMap, dir: std.fs.Dir };
    const pruner_worker = struct {
        fn run(ctx: PrunerCtx) void {
            ctx.map.pruneOrphanedPaths(ctx.dir) catch {};
        }
    }.run;

    var creator_threads: [creator_count]std.Thread = undefined;
    for (0..creator_count) |i| {
        creator_threads[i] = try std.Thread.spawn(.{}, creator_worker, .{CreatorCtx{
            .map = &map,
            .ids = &creator_ids,
            .thread_idx = i,
            .ops = creator_ops,
        }});
    }
    var t_read = try std.Thread.spawn(.{}, reader_worker, .{ReaderCtx{ .map = &map }});
    var t_remove = try std.Thread.spawn(.{}, remover_worker, .{RemoverCtx{ .map = &map }});
    var t_rename = try std.Thread.spawn(.{}, renamer_worker, .{RenamerCtx{ .map = &map }});
    var t_prune = try std.Thread.spawn(.{}, pruner_worker, .{PrunerCtx{ .map = &map, .dir = tmpD.dir }});

    for (&creator_threads) |*t| t.join();
    t_read.join();
    t_remove.join();
    t_rename.join();
    t_prune.join();

    // All creator paths are unique so their assigned IDs must be distinct.
    // A race on next_id causes different paths to receive the same ID.
    for (0..creator_ids.len) |i| {
        for (i + 1..creator_ids.len) |j| {
            try expect(creator_ids[i] != creator_ids[j]);
        }
    }
}

test "pruneOrphanedPaths removes missing files" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    (try tmpD.dir.createFile("exists.md", .{})).close();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    _ = try map.getOrCreateId("exists.md");
    _ = try map.getOrCreateId("missing.md");

    try expectEqual(@as(usize, 2), map.count());

    try map.pruneOrphanedPaths(tmpD.dir);

    try expectEqual(@as(usize, 1), map.count());
    try expect(map.getId("exists.md") != null);
    try expect(map.getId("missing.md") == null);
}

const db_suffix = @import("vec_util.zig").db_suffix;
const std = @import("std");
const tracy = @import("tracy");

fn readIntLE(reader: *std.fs.File.Reader, comptime T: type) !T {
    var buf: [@sizeOf(T)]u8 = undefined;
    try reader.interface.readSliceAll(&buf);
    return std.mem.readInt(T, &buf, .little);
}

fn writeIntLE(writer: *std.fs.File.Writer, comptime T: type, value: T) !void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, value, .little);
    try writer.interface.writeAll(&buf);
}

test "reroot deeper prefixes every path and keeps the ids" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    const one = try map.getOrCreateId("one.md");
    const two = try map.getOrCreateId("sub/two.md");

    var dropped: std.ArrayList(NoteID) = .{};
    defer dropped.deinit(testing_allocator);
    try map.reroot(.{ .deeper = "md-src" }, &dropped);

    // The point of the operation: the vectors never move, so the ids must not either.
    try expectEqual(@as(usize, 0), dropped.items.len);
    try expectEqual(one, map.getId("md-src/one.md").?);
    try expectEqual(two, map.getId("md-src/sub/two.md").?);
    try expectEqual(@as(?NoteID, null), map.getId("one.md"));
    try expectEqualStrings("md-src/one.md", map.getPath(one).?);
    try expectEqual(@as(usize, 2), map.count());
}

test "reroot shallower strips the prefix and reports what fell outside" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    const inside = try map.getOrCreateId("md-src/one.md");
    const deeper = try map.getOrCreateId("md-src/sub/two.md");
    const outside = try map.getOrCreateId("elsewhere/three.md");
    // A name that merely starts with the same letters is not inside the folder.
    const lookalike = try map.getOrCreateId("md-srcery/four.md");

    var dropped: std.ArrayList(NoteID) = .{};
    defer dropped.deinit(testing_allocator);
    try map.reroot(.{ .shallower = "md-src" }, &dropped);

    try expectEqual(inside, map.getId("one.md").?);
    try expectEqual(deeper, map.getId("sub/two.md").?);
    try expectEqual(@as(usize, 2), map.count());

    try expectEqual(@as(usize, 2), dropped.items.len);
    var saw_outside = false;
    var saw_lookalike = false;
    for (dropped.items) |id| {
        if (id == outside) saw_outside = true;
        if (id == lookalike) saw_lookalike = true;
    }
    try expect(saw_outside and saw_lookalike);
    try expectEqual(@as(?[]const u8, null), map.getPath(outside));
}

test "reroot survives a restart" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var id: NoteID = undefined;
    {
        var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
        defer map.deinit();
        id = try map.getOrCreateId("one.md");

        var dropped: std.ArrayList(NoteID) = .{};
        defer dropped.deinit(testing_allocator);
        try map.reroot(.{ .deeper = "md-src" }, &dropped);
    }
    {
        // The manifest on disk is what the next launch reads, so the rewrite has to be in it
        // and not just in the map that performed it.
        var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
        defer map.deinit();
        try expectEqual(id, map.getId("md-src/one.md").?);
        try expectEqual(@as(usize, 1), map.count());
    }
}

test "rerooting an empty map is not an error" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var map = try NoteIdMap.init(testing_allocator, tmpD.dir);
    defer map.deinit();

    var dropped: std.ArrayList(NoteID) = .{};
    defer dropped.deinit(testing_allocator);
    try map.reroot(.{ .deeper = "md-src" }, &dropped);
    try expectEqual(@as(usize, 0), map.count());
}
