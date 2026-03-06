const std = @import("std");
const dve = @import("dve");

const VectorDB = dve.VectorDB(dve.embedding_model);

const DOCUMENTS = [_]struct { key: []const u8, text: []const u8 }{
    .{ .key = "solar-system", .text = "The solar system consists of the Sun and the objects that orbit it, including eight planets, their moons, and countless asteroids and comets." },
    .{ .key = "photosynthesis", .text = "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose." },
    .{ .key = "machine-learning", .text = "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed." },
    .{ .key = "ancient-rome", .text = "Ancient Rome was a civilization that grew from a small agricultural community on the Italian Peninsula into a vast empire spanning much of Europe, the Middle East, and North Africa." },
    .{ .key = "quantum-mechanics", .text = "Quantum mechanics is a fundamental theory in physics that describes the behavior of nature at the smallest scales, where particles can exist in multiple states simultaneously." },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Use a temp directory for the database, cleaned up on exit.
    const tmp_path = "/tmp/dve-repl";
    std.fs.deleteTreeAbsolute(tmp_path) catch {};
    var tmp_dir = try std.fs.cwd().makeOpenPath(tmp_path, .{});
    defer {
        tmp_dir.close();
        std.fs.deleteTreeAbsolute(tmp_path) catch {};
    }

    var embedder = try dve.embed.NLEmbedder.init();
    const db = try VectorDB.init(allocator, tmp_dir, embedder.embedder());
    defer db.deinit();

    // Embed all documents.
    std.debug.print("Embedding documents...\n\n", .{});
    for (DOCUMENTS) |doc| {
        try db.embedText(doc.key, doc.text);
        std.debug.print("  [{s}]\n  {s}\n\n", .{ doc.key, doc.text });
    }

    // Query loop.
    const stdin = std.io.getStdIn().reader();
    var buf: [256]u8 = undefined;

    while (true) {
        std.debug.print("Query (or 'quit'): ", .{});
        const line = try stdin.readUntilDelimiterOrEof(&buf, '\n') orelse break;
        const query = std.mem.trimRight(u8, line, "\r\n");
        if (std.mem.eql(u8, query, "quit")) break;
        if (query.len == 0) continue;

        var results: [5]dve.SearchResult = undefined;
        const n = try db.search(query, &results);

        if (n == 0) {
            std.debug.print("No results.\n\n", .{});
            continue;
        }

        for (results[0..n]) |result| {
            std.debug.print("  [{s}] (similarity: {d:.2})\n", .{ result.path, result.similarity });
        }
        std.debug.print("\n", .{});
    }
}
