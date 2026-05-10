const std = @import("std");
const dve = @import("dve");

const VectorEngine = dve.VectorEngine(dve.embedding_model);

const OPS_PER_WORKER: u32 = 1;
const MAX_KEY_LEN: usize = 256;
const MAX_CONTENT_LEN: usize = 8192;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var is_worker = false;
    var stop_on_fail = false;
    var seed: u64 = std.crypto.random.int(u64);
    var max_iterations: ?u64 = null;

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
        } else if (std.mem.eql(u8, args[i], "--stop-on-fail")) {
            stop_on_fail = true;
        }
    }

    if (is_worker) {
        try runWorker(allocator, seed);
    } else {
        try runCoordinator(allocator, seed, max_iterations, stop_on_fail);
    }
}

fn help() void {
    std.debug.print("usage: dve-fuzz [--worker] [--seed N] [--stop-on-fail] [--max-iterations N]\n", .{});
}

fn writeOut(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(s);
    try std.fs.File.stdout().writeAll(s);
}

fn runCoordinator(
    allocator: std.mem.Allocator,
    initial_seed: u64,
    max_iterations: ?u64,
    stop_on_fail: bool,
) !void {
    var prng = std.Random.DefaultPrng.init(initial_seed);
    const rand = prng.random();

    const exe_path = try std.fs.selfExePathAlloc(allocator);
    defer allocator.free(exe_path);

    var i: u64 = 0;
    while (true) {
        const worker_seed = rand.int(u64);
        const seed_str = try std.fmt.allocPrint(allocator, "{d}", .{worker_seed});
        defer allocator.free(seed_str);

        var child = std.process.Child.init(
            &.{ exe_path, "--worker", "--seed", seed_str },
            allocator,
        );
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Inherit;
        try child.spawn();

        var buf: [4096]u8 = undefined;
        while (true) {
            const n = try child.stdout.?.read(&buf);
            if (n == 0) break;
            try std.fs.File.stdout().writeAll(buf[0..n]);
        }

        const term = try child.wait();
        const crashed = switch (term) {
            .Exited => |code| code != 0,
            .Signal, .Stopped, .Unknown => true,
        };

        if (crashed) {
            try writeOut(allocator, "{{\"event\":\"crash\",\"seed\":{d}}}\n", .{worker_seed});
            if (stop_on_fail) break;
        }
        if (max_iterations != null and i >= max_iterations.?) break;
        i += 1;
    }
}

const Op = enum { embed, embedAsync, search, uniqueSearch, populateHighlights, remove, rename };

fn runWorker(allocator: std.mem.Allocator, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const tmp_path = try std.fmt.allocPrint(allocator, "/tmp/dve-fuzz-{d}", .{seed});
    defer allocator.free(tmp_path);
    std.fs.deleteTreeAbsolute(tmp_path) catch {};
    try std.fs.makeDirAbsolute(tmp_path);
    defer std.fs.deleteTreeAbsolute(tmp_path) catch {};

    var tmp_dir = try std.fs.openDirAbsolute(tmp_path, .{});
    defer tmp_dir.close();

    try writeOut(allocator, "{{\"event\":\"start\",\"seed\":{d}}}\n", .{seed});

    var embedder = try dve.embed.MpnetEmbedder.init(.{});
    const engine = try VectorEngine.init(allocator, tmp_dir, embedder.embedder());
    defer engine.deinit();

    var key_buf: [MAX_KEY_LEN]u8 = undefined;
    var key2_buf: [MAX_KEY_LEN]u8 = undefined;
    var content_buf: [MAX_CONTENT_LEN]u8 = undefined;
    var content2_buf: [MAX_CONTENT_LEN]u8 = undefined;
    var search_results: [10]dve.SearchResult = undefined;
    var highlights_buf: [20]usize = undefined;

    var op_i: u32 = 0;
    while (op_i < OPS_PER_WORKER) : (op_i += 1) {
        const op = @as(Op, @enumFromInt(rand.uintLessThan(u8, std.meta.fields(Op).len)));

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
                    "{{\"event\":\"attempt\",\"op\":\"embed\",\"key\":\"{s}\",\"content\":\"{s}\"}}\n",
                    .{ key_json, content_json },
                );
                engine.embedText(key, content) catch |err| {
                    try writeOut(allocator, "{{\"event\":\"error\",\"op\":\"embed\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, "{{\"event\":\"ok\",\"op\":\"embed\"}}\n", .{});
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
                    "{{\"event\":\"attempt\",\"op\":\"embedAsync\",\"key\":\"{s}\",\"content\":\"{s}\"}}\n",
                    .{ key_json, content_json },
                );
                engine.embedTextAsync(key, content) catch |err| {
                    try writeOut(allocator, "{{\"event\":\"error\",\"op\":\"embedAsync\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, "{{\"event\":\"ok\",\"op\":\"embedAsync\"}}\n", .{});
            },
            .search => {
                const query = randString(rand, &key_buf);
                const query_json = try jsonEscape(allocator, query);
                defer allocator.free(query_json);
                try writeOut(
                    allocator,
                    "{{\"event\":\"attempt\",\"op\":\"search\",\"query\":\"{s}\"}}\n",
                    .{query_json},
                );
                const n = engine.search(query, &search_results) catch |err| {
                    try writeOut(allocator, "{{\"event\":\"error\",\"op\":\"search\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, "{{\"event\":\"ok\",\"op\":\"search\",\"n\":{d}}}\n", .{n});
            },
            .uniqueSearch => {
                const query = randString(rand, &key_buf);
                const query_json = try jsonEscape(allocator, query);
                defer allocator.free(query_json);
                try writeOut(
                    allocator,
                    "{{\"event\":\"attempt\",\"op\":\"uniqueSearch\",\"query\":\"{s}\"}}\n",
                    .{query_json},
                );
                const n = engine.uniqueSearch(query, &search_results) catch |err| {
                    try writeOut(allocator, "{{\"event\":\"error\",\"op\":\"uniqueSearch\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, "{{\"event\":\"ok\",\"op\":\"uniqueSearch\",\"n\":{d}}}\n", .{n});
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
                    "{{\"event\":\"attempt\",\"op\":\"populateHighlights\",\"query\":\"{s}\",\"content\":\"{s}\"}}\n",
                    .{ query_json, content_json },
                );
                engine.populateHighlights(query, content, &highlights_buf) catch |err| {
                    try writeOut(allocator, "{{\"event\":\"error\",\"op\":\"populateHighlights\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, "{{\"event\":\"ok\",\"op\":\"populateHighlights\"}}\n", .{});
            },
            .remove => {
                const key = randString(rand, &key_buf);
                const key_json = try jsonEscape(allocator, key);
                defer allocator.free(key_json);
                try writeOut(
                    allocator,
                    "{{\"event\":\"attempt\",\"op\":\"remove\",\"key\":\"{s}\"}}\n",
                    .{key_json},
                );
                engine.removePath(key) catch |err| {
                    try writeOut(allocator, "{{\"event\":\"error\",\"op\":\"remove\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, "{{\"event\":\"ok\",\"op\":\"remove\"}}\n", .{});
            },
            .rename => {
                const old_key = randString(rand, &key_buf);
                const new_key = randString(rand, &key2_buf);
                const old_json = try jsonEscape(allocator, old_key);
                defer allocator.free(old_json);
                const new_json = try jsonEscape(allocator, new_key);
                defer allocator.free(new_json);
                try writeOut(
                    allocator,
                    "{{\"event\":\"attempt\",\"op\":\"rename\",\"old_key\":\"{s}\",\"new_key\":\"{s}\"}}\n",
                    .{ old_json, new_json },
                );
                engine.renamePath(old_key, new_key) catch |err| {
                    try writeOut(allocator, "{{\"event\":\"error\",\"op\":\"rename\",\"err\":\"{s}\"}}\n", .{@errorName(err)});
                    continue;
                };
                try writeOut(allocator, "{{\"event\":\"ok\",\"op\":\"rename\"}}\n", .{});
            },
        }
    }

    try writeOut(allocator, "{{\"event\":\"done\",\"ops\":{d}}}\n", .{op_i});
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
