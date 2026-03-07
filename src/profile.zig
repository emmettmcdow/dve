const std = @import("std");
const dve = @import("dve");
const embed = dve.embed;
const embedding_model = dve.embedding_model;
const NoteID = dve.note_id_map.NoteID;

const testing_allocator = std.testing.allocator;

const VEC_SZ: usize = switch (embedding_model) {
    .apple_nlembedding => embed.NLEmbedder.VEC_SZ,
    .mpnet_embedding => embed.MpnetEmbedder.VEC_SZ,
};
const VEC_TYPE = f32;
const Vector = @Vector(VEC_SZ, VEC_TYPE);

const Embedder = switch (embedding_model) {
    .apple_nlembedding => embed.NLEmbedder,
    .mpnet_embedding => embed.MpnetEmbedder,
};

const TestVecDB = dve.VectorEngine(embedding_model);
const VecStorage = dve.vec_storage.Storage(VEC_SZ, VEC_TYPE);

const words = [_][]const u8{
    // tech
    "algorithm", "database", "network",      "server",       "client",    "cache",       "memory",      "processor",
    "software",  "hardware", "kernel",        "compiler",     "runtime",   "library",     "framework",   "protocol",
    "encryption","authentication","deployment","containerize",
    // nature
    "forest",    "mountain", "river",         "ocean",        "desert",    "meadow",      "canyon",      "glacier",
    "waterfall", "volcano",  "island",        "peninsula",    "tundra",    "savanna",     "rainforest",  "estuary",
    "boulder",   "pebble",   "horizon",       "twilight",
    // animals
    "eagle",     "salmon",   "wolf",          "dolphin",      "elephant",  "cheetah",     "crocodile",   "penguin",
    "octopus",   "butterfly","jaguar",        "mongoose",     "narwhal",   "platypus",    "quokka",      "falcon",
    "gecko",     "ibis",     "lynx",          "marmot",
    // food
    "bread",     "cheese",   "mango",         "avocado",      "pasta",     "sushi",       "curry",       "noodle",
    "soup",      "salad",    "coffee",        "chocolate",    "almond",    "blueberry",   "pumpkin",     "fennel",
    "turmeric",  "tahini",   "kimchi",        "tempeh",
    // verbs
    "explore",   "discover", "build",         "analyze",      "create",    "transform",   "optimize",    "integrate",
    "collaborate","innovate","migrate",       "deploy",       "monitor",   "evaluate",    "generate",    "simulate",
    "iterate",   "validate", "benchmark",     "profile",
    // adjectives
    "efficient", "scalable", "robust",        "elegant",      "complex",   "dynamic",     "static",      "parallel",
    "distributed","autonomous","resilient",   "flexible",     "modular",   "portable",    "reliable",    "secure",
    "immutable", "concurrent","asynchronous", "deterministic",
    // abstract
    "concept",   "theory",   "pattern",       "principle",    "paradigm",  "abstraction", "entropy",     "complexity",
    "convergence","divergence","recursion",   "iteration",    "transformation","emergence","coherence",  "balance",
    "threshold", "boundary", "gradient",      "dimension",
    // weather / society
    "thunder",   "lightning","blizzard",      "drought",      "monsoon",   "tornado",     "hurricane",   "rainbow",
    "culture",   "society",  "economy",       "technology",   "civilization","community", "heritage",    "tradition",
    "pressure",  "humidity", "climate",       "innovation",
};

fn generateNote(buf: []u8, rng: std.Random, term_idx: *usize) usize {
    const terminators = [_][]const u8{ ". ", "! ", "? " };
    var pos: usize = 0;
    const sentence_count = rng.intRangeAtMost(usize, 5, 10);
    for (0..sentence_count) |_| {
        const word_count = rng.intRangeAtMost(usize, 5, 8);
        for (0..word_count) |wi| {
            if (pos >= buf.len) return pos;
            if (wi > 0) {
                buf[pos] = ' ';
                pos += 1;
                if (pos >= buf.len) return pos;
            }
            const word = words[rng.intRangeLessThan(usize, 0, words.len)];
            const to_copy = @min(word.len, buf.len - pos);
            @memcpy(buf[pos .. pos + to_copy], word[0..to_copy]);
            pos += to_copy;
        }
        const term = terminators[term_idx.* % terminators.len];
        term_idx.* += 1;
        const to_copy = @min(term.len, buf.len - pos);
        @memcpy(buf[pos .. pos + to_copy], term[0..to_copy]);
        pos += to_copy;
    }
    return pos;
}

fn testEmbedder(allocator: std.mem.Allocator) !struct { e: *Embedder, iface: embed.Embedder } {
    const e = try allocator.create(Embedder);
    if (embedding_model == .mpnet_embedding) {
        e.* = try embed.MpnetEmbedder.init(.{});
    } else {
        e.* = try embed.NLEmbedder.init();
    }
    return .{ .e = e, .iface = e.embedder() };
}

fn randomUnitVector(rng: std.Random) Vector {
    var v: Vector = @splat(0.0);
    var sum_sq: f32 = 0.0;
    for (0..VEC_SZ) |i| {
        const val = rng.float(f32) * 2.0 - 1.0;
        v[i] = val;
        sum_sq += val * val;
    }
    const norm: Vector = @splat(@sqrt(sum_sq));
    return v / norm;
}

const NOTE_COUNT = 100;
const NOTE_BUF_SIZE = 512;
const SEARCH_QUERIES = 20;

test "profile embedding" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();

    const te = try testEmbedder(testing_allocator);
    defer testing_allocator.destroy(te.e);
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, te.iface);
    defer db.deinit();

    var path_buf: [32]u8 = undefined;
    var note_buf: [NOTE_BUF_SIZE]u8 = undefined;
    var term_idx: usize = 0;
    var embed_ns: u64 = 0;
    var total_sentences: usize = 0;

    // === EMBEDDING ===
    for (0..NOTE_COUNT) |i| {
        const note_len = generateNote(&note_buf, rng, &term_idx);
        const note = note_buf[0..note_len];
        for (note) |c| {
            if (c == '.' or c == '!' or c == '?') total_sentences += 1;
        }
        const path = std.fmt.bufPrint(&path_buf, "note_{d}", .{i}) catch unreachable;
        const t0: i128 = std.time.nanoTimestamp();
        try db.embedText(path, note);
        const t1: i128 = std.time.nanoTimestamp();
        embed_ns += @intCast(t1 - t0);
    }

    const embed_s = @as(f64, @floatFromInt(embed_ns)) / 1e9;
    const ms_per_sentence = if (total_sentences > 0)
        @as(f64, @floatFromInt(embed_ns)) / 1e6 / @as(f64, @floatFromInt(total_sentences))
    else
        0.0;
    const sentences_per_sec = if (embed_ns > 0)
        @as(f64, @floatFromInt(total_sentences)) / embed_s
    else
        0.0;

    std.debug.print(
        \\
        \\=== EMBEDDING ===
        \\sentences:        {d}
        \\notes:            {d}
        \\total time:       {d:.1}s
        \\avg per sentence: {d:.1}ms
        \\sentences/sec:    {d:.1}
        \\
    , .{ total_sentences, NOTE_COUNT, embed_s, ms_per_sentence, sentences_per_sec });

    // === SEARCH (small corpus) ===
    var search_buf: [50]dve.SearchResult = undefined;
    var search_ns: u64 = 0;

    for (0..SEARCH_QUERIES) |qi| {
        const query_word = words[rng.intRangeLessThan(usize, 0, words.len)];
        _ = qi;
        const t0: i128 = std.time.nanoTimestamp();
        _ = try db.uniqueSearch(query_word, &search_buf);
        const t1: i128 = std.time.nanoTimestamp();
        search_ns += @intCast(t1 - t0);
    }

    const ms_per_query = @as(f64, @floatFromInt(search_ns)) / 1e6 / @as(f64, SEARCH_QUERIES);

    std.debug.print(
        \\
        \\=== SEARCH (small, ~{d} vecs) ===
        \\queries:         {d}
        \\avg per query:   {d:.2}ms
        \\
    , .{ total_sentences, SEARCH_QUERIES, ms_per_query });
}

test "profile search - large corpus" {
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();

    std.debug.print(
        \\
        \\=== SEARCH (large corpus) ===
        \\
    , .{});

    const sizes = [_]usize{ 1000, 5000, 10000 };
    inline for (sizes) |N| {
        var tmpD = std.testing.tmpDir(.{ .iterate = true });
        defer tmpD.cleanup();

        var storage = try VecStorage.init(testing_allocator, tmpD.dir, .{ .sz = N });
        defer storage.deinit();

        // Fill with random unit vectors (no CoreML)
        for (0..N) |i| {
            const vec = randomUnitVector(rng);
            _ = try storage.put(@intCast(i), 0, 1, vec);
        }

        var search_buf: [50]VecStorage.SearchEntry = undefined;
        var search_ns: u64 = 0;

        for (0..SEARCH_QUERIES) |_| {
            const query = randomUnitVector(rng);
            const t0: i128 = std.time.nanoTimestamp();
            _ = try storage.search(query, &search_buf, 0.5);
            const t1: i128 = std.time.nanoTimestamp();
            search_ns += @intCast(t1 - t0);
        }

        const ms_per_query = @as(f64, @floatFromInt(search_ns)) / 1e6 / @as(f64, SEARCH_QUERIES);
        std.debug.print("N={d:<6}  avg: {d:.2}ms\n", .{ N, ms_per_query });
    }
}
