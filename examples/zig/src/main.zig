const std = @import("std");
const dve = @import("dve");

const VectorDB = dve.VectorDB(dve.embedding_model);

const POLL_INTERVAL_NS = 1 * std.time.ns_per_s;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        printUsage();
        std.process.exit(1);
    }

    const subcmd = args[1];
    const dir_path = args[2];

    if (std.mem.eql(u8, subcmd, "watch")) {
        try cmdWatch(allocator, dir_path);
    } else if (std.mem.eql(u8, subcmd, "search")) {
        if (args.len < 4) {
            printUsage();
            std.process.exit(1);
        }
        try cmdSearch(allocator, dir_path, args[3]);
    } else {
        printUsage();
        std.process.exit(1);
    }
}

fn cmdWatch(allocator: std.mem.Allocator, dir_path: []const u8) !void {
    var dir = try std.fs.cwd().makeOpenPath(dir_path, .{ .iterate = true });
    defer dir.close();

    var embedder = try dve.embed.NLEmbedder.init();
    const db = try VectorDB.init(allocator, dir, embedder.embedder());
    defer db.deinit();

    // Track mtimes to detect changes. Map keys are owned (duped) strings.
    var mtimes = std.StringHashMap(i128).init(allocator);
    defer {
        var key_it = mtimes.keyIterator();
        while (key_it.next()) |key| allocator.free(key.*);
        mtimes.deinit();
    }

    std.debug.print("Watching {s} for changes...\n", .{dir_path});

    while (true) {
        var walker = try dir.walk(allocator);
        defer walker.deinit();

        while (try walker.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!isTextFile(entry.basename)) continue;

            const f = dir.openFile(entry.path, .{}) catch continue;
            const mtime = (f.metadata() catch {
                f.close();
                continue;
            }).modified();
            f.close();

            const prev = mtimes.get(entry.path);
            if (prev != null and prev.? == mtime) continue;

            const contents = dir.readFileAlloc(allocator, entry.path, MAX_FILE_SIZE) catch |err| {
                std.debug.print("Failed to read {s}: {}\n", .{ entry.path, err });
                continue;
            };
            defer allocator.free(contents);

            db.embedTextAsync(entry.path, contents) catch |err| {
                std.debug.print("Failed to embed {s}: {}\n", .{ entry.path, err });
                continue;
            };
            std.debug.print("Embedded: {s}\n", .{entry.path});

            if (prev == null) {
                try mtimes.put(try allocator.dupe(u8, entry.path), mtime);
            } else {
                mtimes.getPtr(entry.path).?.* = mtime;
            }
        }

        std.time.sleep(POLL_INTERVAL_NS);
    }
}

fn cmdSearch(allocator: std.mem.Allocator, dir_path: []const u8, query: []const u8) !void {
    var dir = try std.fs.cwd().openDir(dir_path, .{});
    defer dir.close();

    var embedder = try dve.embed.NLEmbedder.init();
    const db = try VectorDB.init(allocator, dir, embedder.embedder());
    defer db.deinit();

    var results: [20]dve.SearchResult = undefined;
    const n = try db.search(query, &results);

    if (n == 0) {
        std.debug.print("No results found.\n", .{});
        return;
    }

    for (results[0..n]) |result| {
        std.debug.print("{s} (similarity: {d:.2})\n", .{ result.path, result.similarity });

        const text_len = result.end_i - result.start_i;
        const buf = try allocator.alloc(u8, text_len);
        defer allocator.free(buf);

        if (dir.openFile(result.path, .{})) |f| {
            defer f.close();
            f.seekTo(result.start_i) catch continue;
            const n_read = f.read(buf) catch continue;
            std.debug.print("   matched text: {s}\n", .{buf[0..n_read]});
        } else |_| continue;
    }
}

fn isTextFile(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".md");
}

fn printUsage() void {
    std.debug.print(
        \\Usage:
        \\  embed-watch watch  <dir>           Watch a directory and embed text files
        \\  embed-watch search <dir> <query>   Search the embedded database
        \\
    , .{});
}
