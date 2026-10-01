const TextEntry = struct { path: []const u8, contents: []const u8 };

const t1 = "binary single words";
fn binarySingleWords(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t1, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try dve.VectorEngine(model).init(testing_allocator, tmpD.dir, .{});
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
    var db = try dve.VectorEngine(model).init(testing_allocator, tmpD.dir, .{});
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
    var db = try dve.VectorEngine(model).init(testing_allocator, tmpD.dir, .{});
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
    var db = try dve.VectorEngine(model).init(testing_allocator, tmpD.dir, .{});
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
    var db = try dve.VectorEngine(model).init(testing_allocator, tmpD.dir, .{});
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

const t6 = "markdown structure";
/// The chunker splits on ".!?\n", so in markdown every list item, table row and front-matter
/// field becomes its own "sentence" -- usually two or three words long. Nothing else in this
/// file has a newline in it, so none of that is exercised anywhere, even though markdown is
/// the format dve is pointed at.
fn markdownStructure(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t6, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try dve.VectorEngine(model).init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const all_docs = [_]TextEntry{
        // The answer is one item in a list.
        .{ .path = "recipes", .contents =
            \\# Recipes
            \\- Pancakes: whisk flour, milk and eggs, then fry in butter
            \\- Risotto: toast arborio rice, then add hot stock a ladle at a time
            \\- Chili: brown the beef, add beans and tomatoes, simmer an hour
        },
        // The answer is one row of a table.
        .{ .path = "ports", .contents =
            \\# Default ports
            \\| service | port |
            \\| ssh | 22 |
            \\| https | 443 |
            \\| postgres | 5432 |
        },
        // The heading is the topic; the answer is in the body under it.
        .{ .path = "deploy", .contents =
            \\## Shipping a release
            \\Bump the version in build.zig, then run the release script from a clean tree.
        },
        // YAML front matter, then the sentence that actually answers anything.
        .{ .path = "meeting", .contents =
            \\---
            \\title: Q3 planning
            \\attendees: Dana, Sam
            \\---
            \\We agreed to delay the launch until October so QA has time.
        },
        // One real sentence buried in a pile of two-word stubs, which is what 82% of the
        // Wikipedia corpus looks like after chunking.
        .{ .path = "biology", .contents =
            \\# Links
            \\- cells
            \\- organelles
            \\- enzymes
            \\- proteins
            \\- lipids
            \\Mitochondria generate most of the chemical energy a cell needs to survive.
        },
    };
    for (all_docs) |doc| try db.embedText(doc.path, doc.contents);

    const cases = [_]SentenceCase{
        .{ .query = "how do I cook risotto", .to_include = &.{"recipes"}, .no_include = &.{ "ports", "meeting" } },
        .{ .query = "which port does postgres listen on", .to_include = &.{"ports"}, .no_include = &.{ "recipes", "biology" } },
        .{ .query = "how do I ship a release", .to_include = &.{"deploy"}, .no_include = &.{ "recipes", "ports" } },
        .{ .query = "when is the launch happening", .to_include = &.{"meeting"}, .no_include = &.{ "ports", "biology" } },
        .{ .query = "what do mitochondria do in a cell", .to_include = &.{"biology"}, .no_include = &.{ "ports", "deploy" } },
    };

    var searchBuf: [10]SearchResult = undefined;
    const case_weight = 10;
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

const t7 = "distractor density";
/// The failure this file could not previously see.
///
/// On the 1.18M-vector Wikipedia index, "capital city of france" returned a baseball league
/// and a handful of French hamlets, all bunched at 0.80-0.82 similarity, with Paris nowhere --
/// and scored a perfect 1.000 against an exhaustive exact-cosine scan, because the index had
/// faithfully returned the best-scoring vectors there were. A corpus of five documents cannot
/// reproduce that: a wrong answer needs somewhere to come from. What it takes is not a million
/// vectors but a few dozen *semantically adjacent* ones.
fn distractorDensity(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t7, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try dve.VectorEngine(model).init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    // Thirty near-misses on the same topic as the question, and one answer.
    const villages = [_][]const u8{
        "Villy",            "Perceneige",   "Nassigny",       "Treteau",
        "Charroux",         "Moulins",      "Domecy",         "Escolives",
        "Chevannes",        "Augy",         "Vincelles",      "Jussy",
        "Gurgy",            "Appoigny",     "Monéteau",       "Chitry",
        "Irancy",           "Lichères",     "Mailly",         "Ouanne",
        "Toucy",            "Pourrain",     "Egleny",         "Dracy",
        "Lindry",           "Charbuy",      "Branches",       "Chemilly",
        "Héry",             "Seignelay",
    };
    var buf: [256]u8 = undefined;
    for (villages, 0..) |v, i| {
        const contents = try std.fmt.bufPrint(
            &buf,
            "{s} is a small commune in the Yonne department in the Bourgogne region of France.",
            .{v},
        );
        var path_buf: [32]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "village-{d}", .{i});
        try db.embedText(path, contents);
    }
    try db.embedText(
        "paris",
        "Paris is the capital and most populous city of France, on the river Seine.",
    );

    // Twenty-five adjacent facts and one answer, on a different topic, so the result is not
    // an artifact of one sentence's phrasing.
    const vitamins = [_][]const u8{
        "Vitamin A deficiency causes night blindness.",
        "Vitamin B1 deficiency causes beriberi.",
        "Vitamin B2 deficiency causes cracked lips.",
        "Vitamin B3 deficiency causes pellagra.",
        "Vitamin B6 deficiency causes anaemia.",
        "Vitamin B9 deficiency causes neural tube defects.",
        "Vitamin B12 deficiency causes pernicious anaemia.",
        "Vitamin D deficiency causes rickets in children.",
        "Vitamin E deficiency causes nerve damage.",
        "Vitamin K deficiency causes excessive bleeding.",
        "Iron deficiency causes fatigue and pallor.",
        "Iodine deficiency causes goitre.",
        "Zinc deficiency causes poor wound healing.",
        "Calcium deficiency causes weak bones.",
        "Magnesium deficiency causes muscle cramps.",
        "Selenium deficiency causes cardiomyopathy.",
        "Copper deficiency causes anaemia and bone problems.",
        "Potassium deficiency causes irregular heartbeat.",
        "Sodium deficiency causes confusion and seizures.",
        "Phosphorus deficiency causes bone pain.",
        "Chromium deficiency affects blood sugar control.",
        "Manganese deficiency affects bone formation.",
        "Fluoride deficiency increases tooth decay.",
        "Choline deficiency causes liver damage.",
        "Biotin deficiency causes hair loss.",
    };
    for (vitamins, 0..) |v, i| {
        var path_buf: [32]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "nutrient-{d}", .{i});
        try db.embedText(path, v);
    }
    try db.embedText(
        "scurvy",
        "Scurvy is a disease caused by a lack of vitamin C in the diet.",
    );

    const Case = struct { query: []const u8, want: []const u8 };
    const cases = [_]Case{
        .{ .query = "what is the capital city of France", .want = "paris" },
        .{ .query = "which vitamin deficiency causes scurvy", .want = "scurvy" },
    };

    var searchBuf: [10]SearchResult = undefined;
    for (cases) |case| {
        const n_out = try db.uniqueSearch(case.query, &searchBuf);

        // Ranked first is the only outcome a user would call correct; in the top three is
        // the weaker claim that the answer at least beat most of the noise. Scored
        // separately so a change that fixes the ordering is visible from a change that only
        // stops the answer being buried.
        curr_max_score += 20;
        if (n_out > 0 and std.mem.eql(u8, searchBuf[0].path, case.want)) curr_score += 20;

        curr_max_score += 10;
        if (outputContains(searchBuf[0..@min(n_out, 3)], case.want)) curr_score += 10;
    }
}

const t8 = "absent answer precision";
/// A query with no answer in the corpus should come back empty, and this is the one place
/// `Embedder.threshold` is actually under test.
///
/// It is not obviously doing its job: on Wikipedia, queries whose answer was missing still
/// returned ten results at 0.80-0.82, well above llama's 0.55 cutoff. Short chunks seem to
/// sit in a region of the space where everything is mildly similar to everything, which makes
/// an absolute threshold a poor filter -- but nothing measured that until now.
fn absentAnswer(comptime model: EmbeddingModel) !void {
    var curr_max_score: usize = 0;
    var curr_score: usize = 0;
    defer reportTest(model, t8, curr_score, curr_max_score);

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try dve.VectorEngine(model).init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const all_docs = [_]TextEntry{
        .{ .path = "1", .contents = "The kettle boils water for tea in about two minutes." },
        .{ .path = "2", .contents = "I replaced the bicycle's rear brake pads on Saturday." },
        .{ .path = "3", .contents = "The sourdough needs a twelve hour bulk ferment." },
        .{ .path = "4", .contents = "Our cat sleeps on the windowsill most afternoons." },
        .{ .path = "5", .contents = "The bus to the station leaves every twenty minutes." },
    };
    for (all_docs) |doc| try db.embedText(doc.path, doc.contents);

    // The corpus needs the stubs, or this measures nothing.
    //
    // On the 2.7M-vector index, "quarterly revenue by region" came back with ten hits at
    // 0.7419-0.7420 -- above llama's 0.55 -- and the offsets gave them away: [466..473],
    // [583..690], [1975..1982], seven and eight byte chunks, nine different documents
    // scoring bit-identically. They are the two-word fragments the ".!?\n" split makes out
    // of markdown lists and infobox fields, and they score ~0.74 against an arbitrary
    // question because a two-word string sits in a part of the space that is mildly close to
    // everything. An absolute threshold cannot separate that, and the first version of this
    // test passed only because its corpus had no stubs in it to find.
    const stub_docs = [_]TextEntry{
        .{ .path = "stubs-1", .contents =
            \\# Index
            \\- Revenue
            \\- Region
            \\- Quarter
            \\- Summary
        },
        .{ .path = "stubs-2", .contents =
            \\# Fields
            \\- Session
            \\- Configure
            \\- Point
            \\- Melting
        },
        .{ .path = "stubs-3", .contents =
            \\| key | value |
            \\| type | none |
            \\| area | n/a |
        },
    };
    for (stub_docs) |doc| try db.embedText(doc.path, doc.contents);

    const absent = [_][]const u8{
        "quarterly revenue by region",
        "symptoms of appendicitis",
        "how to configure a BGP session",
        "the melting point of tungsten",
    };

    var searchBuf: [10]SearchResult = undefined;
    for (absent) |query| {
        const n_out = try db.uniqueSearch(query, &searchBuf);
        // Nothing is the right answer. One stray hit is forgivable; a full page of them means
        // the threshold is not separating anything.
        curr_max_score += 20;
        if (n_out == 0) {
            curr_score += 20;
        } else if (n_out <= 2) {
            curr_score += 10;
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
test "mpnet_embedding: markdown structure" {
    try markdownStructure(.mpnet_embedding);
}
test "mpnet_embedding: distractor density" {
    try distractorDensity(.mpnet_embedding);
}
test "mpnet_embedding: absent answer precision" {
    try absentAnswer(.mpnet_embedding);
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
test "apple_nlembedding: markdown structure" {
    try markdownStructure(.apple_nlembedding);
}
test "apple_nlembedding: distractor density" {
    try distractorDensity(.apple_nlembedding);
}
test "apple_nlembedding: absent answer precision" {
    try absentAnswer(.apple_nlembedding);
}
test "apple_nlembedding: all" {
    reportTotal(.apple_nlembedding);
}

// The llama backend is only linked with -Dllama, so these skip by default
// rather than failing a plain `zig build test`.
test "llama_nomic_embed_text_v1_5_f32: binary single words" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    try binarySingleWords(.llama_nomic_embed_text_v1_5_f32);
}
test "llama_nomic_embed_text_v1_5_f32: sentence similarity" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    try sentenceSimilarity(.llama_nomic_embed_text_v1_5_f32);
}
test "llama_nomic_embed_text_v1_5_f32: sentence split - 1/3 match" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    try sentenceSplit(.llama_nomic_embed_text_v1_5_f32);
}
test "llama_nomic_embed_text_v1_5_f32: query length parity" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    try queryLengthParity(.llama_nomic_embed_text_v1_5_f32);
}
test "llama_nomic_embed_text_v1_5_f32: long complex sentences" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    try longComplexSentences(.llama_nomic_embed_text_v1_5_f32);
}
test "llama_nomic_embed_text_v1_5_f32: markdown structure" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    try markdownStructure(.llama_nomic_embed_text_v1_5_f32);
}
test "llama_nomic_embed_text_v1_5_f32: distractor density" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    try distractorDensity(.llama_nomic_embed_text_v1_5_f32);
}
test "llama_nomic_embed_text_v1_5_f32: absent answer precision" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    try absentAnswer(.llama_nomic_embed_text_v1_5_f32);
}
test "llama_nomic_embed_text_v1_5_f32: all" {
    if (!dve.llama.enabled) return error.SkipZigTest;
    reportTotal(.llama_nomic_embed_text_v1_5_f32);
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
