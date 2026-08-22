const TextEntry = struct { path: []const u8, contents: []const u8 };

const t1 = "binary single words";
fn binarySingleWords(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t1, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try dve.VectorEngine(model).init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    const BiCase = struct { a: []const u8, b: []const u8, query: []const u8, want: []const u8 };
    const binary_cases = [_]BiCase{
        .{ .a = "peasant", .b = "queen", .query = "royalty", .want = "b" },
        .{ .a = "soccer", .b = "sushi", .query = "sport", .want = "a" },
        .{ .a = "world", .b = "calculator", .query = "earth", .want = "a" },
        .{ .a = "night", .b = "day", .query = "moon", .want = "a" },
        .{ .a = "mouse", .b = "dog", .query = "computer", .want = "a" },
        // Additional cases
        .{ .a = "hammer", .b = "paintbrush", .query = "construction", .want = "a" },
        .{ .a = "violin", .b = "trumpet", .query = "strings", .want = "a" },
        .{ .a = "ocean", .b = "desert", .query = "water", .want = "a" },
        .{ .a = "winter", .b = "summer", .query = "cold", .want = "a" },
        .{ .a = "doctor", .b = "lawyer", .query = "medicine", .want = "a" },
    };
    inline for (binary_cases) |case| {
        curr_max_score += 40;
        try db.embedText("a", case.a);
        defer db.removePath("a") catch unreachable;
        try db.embedText("b", case.b);
        defer db.removePath("b") catch unreachable;
        var searchBuf: [10]SearchResult = undefined;
        const n_out = try db.uniqueSearch(case.query, &searchBuf);
        if (n_out > 0) {
            if (std.mem.eql(u8, searchBuf[0].path, case.want)) {
                curr_score += 20;
                if (n_out == 1) curr_score += 20;
            }
        }
    }
}

const SentenceCase = struct {
    query: []const u8,
    to_include: []const []const u8,
    no_include: []const []const u8,
};

const t2 = "sentence similarity";
fn sentenceSimilarity(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t2, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try dve.VectorEngine(model).init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    var searchBuf: [20]SearchResult = undefined;

    const all_docs = [_]TextEntry{
        // Programming-related
        .{ .path = "1", .contents = "Top techniques for mastering coding skills quickly." },
        .{ .path = "2", .contents = "How to improve your skills in software development." },
        .{ .path = "3", .contents = "The ultimate guide to becoming a better programmer" },
        .{ .path = "4", .contents = "Why learning to code is easier with these tips" },
        .{ .path = "5", .contents = "Practice your coding skills" },
        // Food-related
        .{ .path = "6", .contents = "What to eat for a healthy breakfast." },
        .{ .path = "7", .contents = "The best recipes for homemade pasta dishes" },
        .{ .path = "8", .contents = "Nutrition tips for athletes and fitness enthusiasts" },
        // Misc unrelated
        .{ .path = "9", .contents = "She sells sea shells by the sea shore" },
        .{ .path = "10", .contents = "My dog likes to play with other dogs" },
        .{ .path = "11", .contents = "Do you touch type or hunt and peck?" },
        // Travel-related
        .{ .path = "12", .contents = "Best destinations for a summer vacation in Europe" },
        .{ .path = "13", .contents = "How to pack light for international travel" },
        .{ .path = "14", .contents = "Budget tips for backpacking through Asia" },
    };
    for (all_docs) |doc| try db.embedText(doc.path, doc.contents);

    const cases = [_]SentenceCase{
        .{
            .query = "Best strategies for learning programming",
            .to_include = &.{ "1", "2", "3", "4", "5" },
            .no_include = &.{ "6", "7", "8", "9", "10", "11", "12", "13", "14" },
        },
        .{
            .query = "Cooking and meal preparation",
            .to_include = &.{ "6", "7" },
            .no_include = &.{ "1", "2", "3", "4", "5", "9", "10", "11", "12", "13", "14" },
        },
        .{
            .query = "Planning a trip abroad",
            .to_include = &.{ "12", "13", "14" },
            .no_include = &.{ "1", "2", "3", "4", "5", "6", "7", "9", "10", "11" },
        },
    };

    const case_weight: usize = 10; // 40 items × 10 = 400 max
    for (cases) |case| {
        const n_out = try db.uniqueSearch(case.query, &searchBuf);
        for (case.to_include) |path| {
            curr_max_score += case_weight;
            if (outputContains(searchBuf[0..n_out], path)) curr_score += case_weight;
        }
        for (case.no_include) |path| {
            curr_max_score += case_weight;
            if (!outputContains(searchBuf[0..n_out], path)) curr_score += case_weight;
        }
    }
}

const t3 = "sentence split - 1/3 match";
fn sentenceSplit(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t3, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try dve.VectorEngine(model).init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    var searchBuf: [20]SearchResult = undefined;

    const all_docs = [_]TextEntry{
        .{ .path = "1", .contents = "I rode bikes with my friends. We ate hot dogs. Then we went home." },
        .{ .path = "2", .contents = "I graduated college last week. Lots of people had a party. My parents took me to dinner." },
        .{ .path = "3", .contents = "I woke up. I brushed my teeth vigorously! I drove to work." },
        .{ .path = "4", .contents = "The cat slept all day. It played with yarn. Then it ate its dinner." },
        .{ .path = "5", .contents = "We hiked up the mountain. The view was incredible. We took many photos." },
        .{ .path = "6", .contents = "She studied for the exam. Her notes were extensive. The test was difficult." },
    };
    for (all_docs) |doc| try db.embedText(doc.path, doc.contents);

    const cases = [_]SentenceCase{
        .{
            .query = "Eating food",
            .to_include = &.{ "1", "2", "4" },
            .no_include = &.{ "3", "5", "6" },
        },
        .{
            .query = "Physical outdoor activity",
            .to_include = &.{ "1", "5" },
            .no_include = &.{ "3", "6" },
        },
        .{
            .query = "Academic study",
            .to_include = &.{ "2", "6" },
            .no_include = &.{ "1", "3", "4", "5" },
        },
    };

    const case_weight: usize = 25; // 16 items × 25 = 400 max
    for (cases) |case| {
        const n_out = try db.search(case.query, &searchBuf);
        for (case.to_include) |path| {
            curr_max_score += case_weight;
            if (n_out > 0 and outputContains(searchBuf[0..n_out], path)) curr_score += case_weight;
        }
        for (case.no_include) |path| {
            curr_max_score += case_weight;
            if (n_out == 0 or !outputContains(searchBuf[0..n_out], path)) curr_score += case_weight;
        }
    }
}

const t4 = "query length parity";
fn queryLengthParity(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t4, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try dve.VectorEngine(model).init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    var searchBuf: [20]SearchResult = undefined;

    const all_docs = [_]TextEntry{
        .{ .path = "auth", .contents = "User authentication and login system" },
        .{ .path = "database", .contents = "PostgreSQL database connection and query handling" },
        .{ .path = "api", .contents = "REST API endpoints for the web application" },
        .{ .path = "cache", .contents = "Redis caching layer for performance optimization" },
        .{ .path = "logging", .contents = "Application logging and error tracking system" },
    };
    for (all_docs) |doc| try db.embedText(doc.path, doc.contents);

    const QueryCase = struct { query: []const u8, expected: []const u8 };
    const cases = [_]QueryCase{
        // Single word queries
        .{ .query = "authentication", .expected = "auth" },
        .{ .query = "database", .expected = "database" },
        .{ .query = "caching", .expected = "cache" },
        // Short phrase queries (should match same docs as single words)
        .{ .query = "user login authentication", .expected = "auth" },
        .{ .query = "database connection", .expected = "database" },
        .{ .query = "caching performance", .expected = "cache" },
        // Longer queries (should still match correctly)
        .{ .query = "how does user authentication work", .expected = "auth" },
        .{ .query = "setting up database connections and queries", .expected = "database" },
        .{ .query = "implementing a caching layer for better performance", .expected = "cache" },
    };

    const case_weight: usize = 44; // 9 cases × 44 = 396 max (≈400)
    for (cases) |case| {
        curr_max_score += case_weight;
        const n_out = try db.uniqueSearch(case.query, &searchBuf);
        if (n_out > 0 and std.mem.eql(u8, searchBuf[0].path, case.expected)) {
            curr_score += case_weight;
        }
    }
}

const t5 = "long complex sentences";
fn longComplexSentences(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t5, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try dve.VectorEngine(model).init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    var searchBuf: [20]SearchResult = undefined;

    const all_docs = [_]TextEntry{
        // Long technical descriptions (~80-120 chars)
        .{
            .path = "auth_long",
            .contents = "The authentication system uses JWT tokens with refresh capabilities and supports OAuth2 integration for third-party providers",
        },
        .{
            .path = "db_long",
            .contents = "Database connections are pooled using PgBouncer with automatic failover to read replicas when the primary becomes unavailable",
        },
        .{
            .path = "cache_long",
            .contents = "The caching layer implements a write-through strategy with Redis cluster for horizontal scaling and automatic cache invalidation",
        },
        // Short versions of same topics (~30-40 chars)
        .{ .path = "auth_short", .contents = "JWT authentication with OAuth2" },
        .{ .path = "db_short", .contents = "PostgreSQL connection pooling" },
        .{ .path = "cache_short", .contents = "Redis caching with invalidation" },
        // Unrelated documents
        .{ .path = "unrelated1", .contents = "The weather forecast predicts sunny skies and warm temperatures throughout the weekend" },
        .{ .path = "unrelated2", .contents = "Cooking pasta requires boiling water and adding salt before the noodles" },
    };
    for (all_docs) |doc| try db.embedText(doc.path, doc.contents);

    const cases = [_]SentenceCase{
        // Query should match long version (more specific/relevant) over short
        .{
            .query = "How do I set up JWT authentication with OAuth2 providers?",
            .to_include = &.{ "auth_long", "auth_short" },
            .no_include = &.{ "db_long", "db_short", "cache_long", "cache_short", "unrelated1", "unrelated2" },
        },
        .{
            .query = "Database connection pooling and failover configuration",
            .to_include = &.{ "db_long", "db_short" },
            .no_include = &.{ "auth_long", "auth_short", "cache_long", "cache_short", "unrelated1", "unrelated2" },
        },
        .{
            .query = "Redis cache invalidation and scaling strategies",
            .to_include = &.{ "cache_long", "cache_short" },
            .no_include = &.{ "auth_long", "auth_short", "db_long", "db_short", "unrelated1", "unrelated2" },
        },
        // Broader query should still find relevant docs
        .{
            .query = "security and access control",
            .to_include = &.{"auth_long"},
            .no_include = &.{ "unrelated1", "unrelated2" },
        },
    };

    const case_weight: usize = 13; // 31 items × 13 = 403 max (≈400)
    for (cases) |case| {
        const n_out = try db.uniqueSearch(case.query, &searchBuf);
        for (case.to_include) |path| {
            curr_max_score += case_weight;
            if (outputContains(searchBuf[0..n_out], path)) curr_score += case_weight;
        }
        for (case.no_include) |path| {
            curr_max_score += case_weight;
            if (!outputContains(searchBuf[0..n_out], path)) curr_score += case_weight;
        }
    }
}

////////////////////////
// Per-model test set //
////////////////////////
// One test per (case x model), named "<model>: <case>". Every model is compiled
// into the binary, so `-Dtest-filter=<model>` is all it takes to run a single
// model's suite -- including its totals row, which shares the prefix. Names are
// string literals rather than decltests because Zig filters on the test name,
// and a decltest is named after its identifier, not its value.
//
// Declaration order matters: Zig runs tests in the order declared, so each
// model's totals test has to follow that model's cases.

test "mpnet_embedding: binary single words" {
    try binarySingleWords(.mpnet_embedding);
}
test "mpnet_embedding: sentence similarity" {
    try sentenceSimilarity(.mpnet_embedding);
}
test "mpnet_embedding: sentence split - 1/3 match" {
    try sentenceSplit(.mpnet_embedding);
}
test "mpnet_embedding: query length parity" {
    try queryLengthParity(.mpnet_embedding);
}
test "mpnet_embedding: long complex sentences" {
    try longComplexSentences(.mpnet_embedding);
}
test "mpnet_embedding: all" {
    reportTotal(.mpnet_embedding);
}

test "apple_nlembedding: binary single words" {
    try binarySingleWords(.apple_nlembedding);
}
test "apple_nlembedding: sentence similarity" {
    try sentenceSimilarity(.apple_nlembedding);
}
test "apple_nlembedding: sentence split - 1/3 match" {
    try sentenceSplit(.apple_nlembedding);
}
test "apple_nlembedding: query length parity" {
    try queryLengthParity(.apple_nlembedding);
}
test "apple_nlembedding: long complex sentences" {
    try longComplexSentences(.apple_nlembedding);
}
test "apple_nlembedding: all" {
    reportTotal(.apple_nlembedding);
}

/////////////
// Scoring //
/////////////
const t_all = "all";

const Score = struct { got: usize = 0, total: usize = 0 };
var totals = std.EnumArray(EmbeddingModel, Score).initFill(.{});

fn outputContains(output: []SearchResult, path: []const u8) bool {
    for (output) |out_item| {
        if (std.mem.eql(u8, out_item.path, path)) return true;
    }
    return false;
}

fn reportTest(model: EmbeddingModel, label: []const u8, got: usize, total: usize) void {
    report(model, label, got, total);
    const t = totals.getPtr(model);
    t.got += got;
    t.total += total;
}

fn reportTotal(model: EmbeddingModel) void {
    const t = totals.get(model);
    report(model, t_all, t.got, t.total);
}

fn report(model: EmbeddingModel, label: []const u8, got: usize, total: usize) void {
    var buf: [50]u8 = undefined;
    const frac = std.fmt.bufPrint(&buf, "{d} / {d}", .{ got, total }) catch @panic("don't care");
    std.debug.print("{s:<18} | {s:<26} | {s:^13} | {d:.1}% \n", .{
        @tagName(model),
        label,
        frac,
        (@as(f32, @floatFromInt(got)) / @as(f32, @floatFromInt(total))) * 100,
    });
}

const std = @import("std");
const testing_allocator = std.testing.allocator;

const dve = @import("dve");
const embed = dve.embed;
const EmbeddingModel = embed.EmbeddingModel;
const SearchResult = dve.SearchResult;
