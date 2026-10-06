const std = @import("std");
const dve = @import("dve");

const VectorEngine = dve.VectorEngine(.mpnet_embedding);

const DEFAULT_OPS_PER_WORKER: u32 = 1;
const MAX_KEY_LEN: usize = 256;
const MAX_CONTENT_LEN: usize = 8192;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var is_worker = false;
    var stop_on_fail = false;
    var path_constraints = false;
    var seed: u64 = undefined;
    io.random(std.mem.asBytes(&seed));
    var max_iterations: ?u64 = null;
    var ops_per_worker: u32 = DEFAULT_OPS_PER_WORKER;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--worker")) {
            is_worker = true;
        } else if (std.mem.eql(u8, args[i], "--seed") and i + 1 < args.len) {
            i += 1;
            seed = std.fmt.parseInt(u64, args[i], 10) catch |err| {
                std.debug.print("--seed requires a number as an argument\n", .{});
                help();
                return err;
            };
        } else if (std.mem.eql(u8, args[i], "--max-iterations") and i + 1 < args.len) {
            i += 1;
            max_iterations = std.fmt.parseInt(u64, args[i], 10) catch |err| {
                std.debug.print("--max-iterations requires a number as an argument\n", .{});
                help();
                return err;
            };
        } else if (std.mem.eql(u8, args[i], "--ops-per-worker") and i + 1 < args.len) {
            i += 1;
            ops_per_worker = std.fmt.parseInt(u32, args[i], 10) catch |err| {
                std.debug.print("--ops-per-worker requires a number as an argument\n", .{});
                help();
                return err;
            };
        } else if (std.mem.eql(u8, args[i], "--stop-on-fail")) {
            stop_on_fail = true;
        } else if (std.mem.eql(u8, args[i], "--path-constraints")) {
            path_constraints = true;
        }
    }

    if (is_worker) {
        try runWorker(allocator, io, seed, ops_per_worker, path_constraints);
    } else {
        try runCoordinator(allocator, io, seed, max_iterations, stop_on_fail, ops_per_worker, path_constraints);
    }
}

fn help() void {
    std.debug.print("usage: dve-fuzz [--worker] [--seed N] [--stop-on-fail] [--max-iterations N] [--ops-per-worker N] [--path-constraints]\n", .{});
}

fn writeOut(allocator: std.mem.Allocator, io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(s);
    try std.Io.File.stdout().writeStreamingAll(io, s);
}

fn runCoordinator(
    allocator: std.mem.Allocator,
    io: std.Io,
    initial_seed: u64,
    max_iterations: ?u64,
    stop_on_fail: bool,
    ops_per_worker: u32,
    path_constraints: bool,
) !void {
    var prng = std.Random.DefaultPrng.init(initial_seed);
    const rand = prng.random();

    const exe_path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(exe_path);

    var i: u64 = 0;
    while (true) {
        const worker_seed = rand.int(u64);
        const seed_str = try std.fmt.allocPrint(allocator, "{d}", .{worker_seed});
        defer allocator.free(seed_str);
        const ops_str = try std.fmt.allocPrint(allocator, "{d}", .{ops_per_worker});
        defer allocator.free(ops_str);

        var worker_args: std.ArrayList([]const u8) = .empty;
        defer worker_args.deinit(allocator);
        try worker_args.appendSlice(allocator, &.{ exe_path, "--worker", "--seed", seed_str, "--ops-per-worker", ops_str });
        if (path_constraints) try worker_args.append(allocator, "--path-constraints");

        var child = try std.process.spawn(io, .{
            .argv = worker_args.items,
            .stdout = .pipe,
            .stderr = .inherit,
        });

        var buf: [4096]u8 = undefined;
        while (true) {
            const n = child.stdout.?.readStreaming(io, &.{&buf}) catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            try std.Io.File.stdout().writeStreamingAll(io, buf[0..n]);
        }

        const term = try child.wait(io);
        const crashed = switch (term) {
            .exited => |code| code != 0,
            .signal, .stopped, .unknown => true,
        };

        if (crashed) {
            try writeOut(allocator, io, "{{\"event\":\"crash\",\"seed\":{d}}}\n", .{worker_seed});
            if (stop_on_fail) break;
        }
        if (max_iterations != null and i >= max_iterations.?) break;
        i += 1;
    }
}

const Op = enum { embed, embedAsync, search, uniqueSearch, populateHighlights, remove, rename };

fn validate(allocator: std.mem.Allocator, io: std.Io, engine: *VectorEngine) !void {
    engine.validate() catch |err| {
        try writeOut(allocator, io, "{{\"event\":\"validate_fail\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
        return err;
    };
}

fn runWorker(allocator: std.mem.Allocator, io: std.Io, seed: u64, ops_per_worker: u32, path_constraints: bool) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const tmp_path = try std.fmt.allocPrint(allocator, "/tmp/dve-fuzz-{d}", .{seed});
    defer allocator.free(tmp_path);
    const cwd = std.Io.Dir.cwd();
    cwd.deleteTree(io, tmp_path) catch {};
    try std.Io.Dir.createDirAbsolute(io, tmp_path, .default_dir);
    defer cwd.deleteTree(io, tmp_path) catch {};

    var tmp_dir = try std.Io.Dir.openDirAbsolute(io, tmp_path, .{});
    defer tmp_dir.close(io);

    try writeOut(allocator, io, "{{\"event\":\"start\",\"seed\":{d}}}\n", .{seed});

    const engine = try VectorEngine.init(allocator, io, tmp_dir, .{});
    defer engine.deinit();

    // When path_constraints is enabled, tracks successfully embedded keys so
    // that remove/rename are only called when a path is known to exist.
    var known_keys: std.ArrayList([]u8) = .empty;
    defer if (path_constraints) {
        for (known_keys.items) |k| allocator.free(k);
        known_keys.deinit(allocator);
    };

    var key_buf: [MAX_KEY_LEN]u8 = undefined;
    var key2_buf: [MAX_KEY_LEN]u8 = undefined;
    var content_buf: [MAX_CONTENT_LEN]u8 = undefined;
    var content2_buf: [MAX_CONTENT_LEN]u8 = undefined;
    var search_results: [10]dve.SearchResult = undefined;
    var highlights_buf: [20]usize = undefined;

    var op_i: u32 = 0;
    while (op_i < ops_per_worker) : (op_i += 1) {
        const op = @as(Op, @fromBackingInt(@intCast(rand.uintLessThan(u8, @typeInfo(Op).@"enum".field_names.len))));

        switch (op) {
            .embed => {
                const key = randString(rand, &key_buf);
                const content = randString(rand, &content_buf);
                const key_json = try jsonEscape(allocator, key);
                defer allocator.free(key_json);
                const content_json = try jsonEscape(allocator, content);
                defer allocator.free(content_json);
                try writeOut(
                    allocator,
                    io,
                    "{{\"event\":\"attempt\",\"op\":\"embed\",\"key\":\"{s}\",\"content\":\"{s}\"}}\n",
                    .{ key_json, content_json },
                );
                engine.embedText(key, content) catch |err| {
                    try writeOut(allocator, io, "{{\"event\":\"error\",\"op\":\"embed\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                if (path_constraints) try known_keys.append(allocator, try allocator.dupe(u8, key));
                try writeOut(allocator, io, "{{\"event\":\"ok\",\"op\":\"embed\"}}\n", .{});
                try validate(allocator, io, engine);
            },
            .embedAsync => {
                const key = randString(rand, &key_buf);
                const content = randString(rand, &content_buf);
                const key_json = try jsonEscape(allocator, key);
                defer allocator.free(key_json);
                const content_json = try jsonEscape(allocator, content);
                defer allocator.free(content_json);
                try writeOut(
                    allocator,
                    io,
                    "{{\"event\":\"attempt\",\"op\":\"embedAsync\",\"key\":\"{s}\",\"content\":\"{s}\"}}\n",
                    .{ key_json, content_json },
                );
                engine.embedTextAsync(key, content) catch |err| {
                    try writeOut(allocator, io, "{{\"event\":\"error\",\"op\":\"embedAsync\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                if (path_constraints) try known_keys.append(allocator, try allocator.dupe(u8, key));
                try writeOut(allocator, io, "{{\"event\":\"ok\",\"op\":\"embedAsync\"}}\n", .{});
                try validate(allocator, io, engine);
            },
            .search => {
                const query = randString(rand, &key_buf);
                const query_json = try jsonEscape(allocator, query);
                defer allocator.free(query_json);
                try writeOut(
                    allocator,
                    io,
                    "{{\"event\":\"attempt\",\"op\":\"search\",\"query\":\"{s}\"}}\n",
                    .{query_json},
                );
                const n = engine.search(query, &search_results) catch |err| {
                    try writeOut(allocator, io, "{{\"event\":\"error\",\"op\":\"search\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, io, "{{\"event\":\"ok\",\"op\":\"search\",\"n\":{d}}}\n", .{n});
            },
            .uniqueSearch => {
                const query = randString(rand, &key_buf);
                const query_json = try jsonEscape(allocator, query);
                defer allocator.free(query_json);
                try writeOut(
                    allocator,
                    io,
                    "{{\"event\":\"attempt\",\"op\":\"uniqueSearch\",\"query\":\"{s}\"}}\n",
                    .{query_json},
                );
                const n = engine.uniqueSearch(query, &search_results) catch |err| {
                    try writeOut(allocator, io, "{{\"event\":\"error\",\"op\":\"uniqueSearch\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, io, "{{\"event\":\"ok\",\"op\":\"uniqueSearch\",\"n\":{d}}}\n", .{n});
            },
            .populateHighlights => {
                const query = randString(rand, &key_buf);
                const content = randString(rand, &content2_buf);
                const query_json = try jsonEscape(allocator, query);
                defer allocator.free(query_json);
                const content_json = try jsonEscape(allocator, content);
                defer allocator.free(content_json);
                try writeOut(
                    allocator,
                    io,
                    "{{\"event\":\"attempt\",\"op\":\"populateHighlights\",\"query\":\"{s}\",\"content\":\"{s}\"}}\n",
                    .{ query_json, content_json },
                );
                engine.populateHighlights(query, content, &highlights_buf) catch |err| {
                    try writeOut(allocator, io, "{{\"event\":\"error\",\"op\":\"populateHighlights\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, io, "{{\"event\":\"ok\",\"op\":\"populateHighlights\"}}\n", .{});
            },
            .remove => {
                if (path_constraints and known_keys.items.len == 0) continue;
                const key_idx: ?usize = if (path_constraints) rand.uintLessThan(usize, known_keys.items.len) else null;
                const key = if (key_idx) |idx| known_keys.items[idx] else randString(rand, &key_buf);
                const key_json = try jsonEscape(allocator, key);
                defer allocator.free(key_json);
                try writeOut(
                    allocator,
                    io,
                    "{{\"event\":\"attempt\",\"op\":\"remove\",\"key\":\"{s}\"}}\n",
                    .{key_json},
                );
                engine.removePath(key) catch |err| {
                    try writeOut(allocator, io, "{{\"event\":\"error\",\"op\":\"remove\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                if (key_idx) |idx| allocator.free(known_keys.swapRemove(idx));
                try writeOut(allocator, io, "{{\"event\":\"ok\",\"op\":\"remove\"}}\n", .{});
                try validate(allocator, io, engine);
            },
            .rename => {
                if (path_constraints and known_keys.items.len == 0) continue;
                const old_key_idx: ?usize = if (path_constraints) rand.uintLessThan(usize, known_keys.items.len) else null;
                const old_key = if (old_key_idx) |idx| known_keys.items[idx] else randString(rand, &key_buf);
                const new_key = randString(rand, &key2_buf);
                const old_json = try jsonEscape(allocator, old_key);
                defer allocator.free(old_json);
                const new_json = try jsonEscape(allocator, new_key);
                defer allocator.free(new_json);
                try writeOut(
                    allocator,
                    io,
                    "{{\"event\":\"attempt\",\"op\":\"rename\",\"old_key\":\"{s}\",\"new_key\":\"{s}\"}}\n",
                    .{ old_json, new_json },
                );
                engine.renamePath(old_key, new_key) catch |err| {
                    try writeOut(allocator, io, "{{\"event\":\"error\",\"op\":\"rename\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                if (old_key_idx) |idx| {
                    allocator.free(known_keys.swapRemove(idx));
                    try known_keys.append(allocator, try allocator.dupe(u8, new_key));
                }
                try writeOut(allocator, io, "{{\"event\":\"ok\",\"op\":\"rename\"}}\n", .{});
                try validate(allocator, io, engine);
            },
        }
    }

    try writeOut(allocator, io, "{{\"event\":\"done\",\"ops\":{d}}}\n", .{op_i});
}

fn randString(rand: std.Random, buf: []u8) []u8 {
    const len_choices = [_]usize{ 0, 1, 5, 20, 100, 500, buf.len };
    const max_len = len_choices[rand.uintLessThan(usize, len_choices.len)];
    const len = if (max_len == 0) 0 else rand.uintAtMost(usize, @min(max_len, buf.len));
    for (buf[0..len]) |*b| {
        b.* = rand.intRangeAtMost(u8, 32, 126);
    }
    return buf[0..len];
}

fn jsonEscape(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0...8, 11, 12, 14...31, 127 => {
                const hex = "0123456789abcdef";
                try out.appendSlice(allocator, "\\u00");
                try out.append(allocator, hex[(c >> 4) & 0xF]);
                try out.append(allocator, hex[c & 0xF]);
            },
            else => try out.append(allocator, c),
        }
    }
    return out.toOwnedSlice(allocator);
}
