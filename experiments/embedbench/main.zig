//! How fast can we turn Wikipedia text into vectors?
//!
//! Embedding is the slowest step in ingest by a wide margin, so this harness
//! isolates it: no storage, no codes, no engine, just text in and vectors out.
//!
//! It measures two things that are easy to conflate:
//!
//!   1. backend    mpnet (CoreML) against llama.cpp/nomic-embed-text-v1.5.
//!   2. call shape `embed` once per sentence against one `embedBatch` per
//!                 document. This is exactly the before and after of the
//!                 change that moved embedTextInternal onto embedBatch, so
//!                 the delta between the two modes is that change's payoff.
//!
//! Documents are the batch unit on purpose. Real ingest calls embedBatch once
//! per document with that document's surviving sentences, so a benchmark that
//! embedded one flat 50,000-sentence batch would be measuring a call shape the
//! engine never makes.
//!
//!   zig build embedbench -Dllama -Doptimize=ReleaseFast
//!   ./zig-out/bin/embedbench --model llama --docs 200

const std = @import("std");
const dve = @import("dve");
const embed = dve.embed;

pub const std_options: std.Options = .{ .log_level = .err };

const MAX_ARTICLE_BYTES: usize = 4 * 1024 * 1024;

const Model = enum { mpnet, llama, nl };
const Mode = enum { single, batch, both };

const Options = struct {
    corpus: []const u8 = "wikitest/wikidata/md",
    model: Model = .mpnet,
    mode: Mode = .both,
    /// Documents to embed. Sentence count follows from the corpus.
    docs: usize = 200,
    /// Shuffle the article list with this seed before taking `docs` of them. Sorted order
    /// puts asteroid stubs and disambiguation pages first, which are not representative.
    seed: u64 = 1,
    /// Embed this many sentences through the chosen backend before timing anything, to pay
    /// the lazy model load and let the machine settle.
    warmup: usize = 32,
    /// Cap on sentences per document. 0 means no cap.
    max_sentences: usize = 0,
    /// Compare every batched vector against the same string embedded alone and report
    /// the worst deviation, instead of timing anything. A native batch changes the
    /// shape of the matmuls, so the two are not expected to agree bit for bit -- but
    /// they must agree to within float noise. Sequences leaking into each other through
    /// a shared decode would show up here as a cosine well below 1.
    verify: bool = false,
    /// Run batch before single. Whichever shape goes first pays for warming the machine,
    /// so a difference that flips when the order flips is the order, not the shape.
    reverse: bool = false,
    /// Write the selected sentences, one per line, here and exit without embedding.
    /// Lets another tool (llama.cpp's own llama-embedding, say) be pointed at exactly
    /// the text this harness would have embedded.
    dump: ?[]const u8 = null,
};

/// One document's worth of text, already split and filtered the way
/// embedTextInternal filters it: what actually reaches the embedder.
const Doc = struct {
    name: []const u8,
    sentences: [][]const u8,
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    const opts = try parseArgs(args);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const docs = try loadDocs(arena.allocator(), opts);
    if (docs.len == 0) fatal("no usable documents in '{s}'", .{opts.corpus});

    var sentence_n: usize = 0;
    var byte_n: usize = 0;
    for (docs) |d| {
        sentence_n += d.sentences.len;
        for (d.sentences) |s| byte_n += s.len;
    }

    std.debug.print(
        \\corpus     {s}
        \\documents  {d}
        \\sentences  {d} ({d:.1} per document, {d:.1} bytes each)
        \\model      {t}
        \\
        \\
    , .{
        opts.corpus,
        docs.len,
        sentence_n,
        @as(f64, @floatFromInt(sentence_n)) / @as(f64, @floatFromInt(docs.len)),
        @as(f64, @floatFromInt(byte_n)) / @as(f64, @floatFromInt(sentence_n)),
        opts.model,
    });

    if (opts.dump) |path| {
        try dumpSentences(path, docs);
        std.debug.print("wrote {d} sentences to {s}\n", .{ sentence_n, path });
        return;
    }

    switch (opts.model) {
        .mpnet => {
            var m = try embed.MpnetEmbedder.init(.{});
            defer m.deinit();
            var e = m.embedder();
            try run(allocator, &e, docs, opts, byte_n);
        },
        .nl => {
            var m = try embed.NLEmbedder.init();
            defer m.deinit();
            var e = m.embedder();
            try run(allocator, &e, docs, opts, byte_n);
        },
        .llama => {
            if (!dve.llama.enabled) fatal("built without -Dllama", .{});
            var m = try embed.LlamaNomicEmbedTextV15F32.init();
            defer m.deinit();
            var e = m.embedder();
            try run(allocator, &e, docs, opts, byte_n);
        },
    }
}

fn run(
    allocator: std.mem.Allocator,
    e: *embed.Embedder,
    docs: []const Doc,
    opts: Options,
    byte_n: usize,
) !void {
    try warmup(allocator, e, docs, opts.warmup);

    if (opts.verify) return verify(allocator, e, docs);

    const header = "mode     docs   sentences      wall(s)    sent/s     KB/s    ms/sent";
    std.debug.print("{s}\n", .{header});
    for (0..header.len) |_| std.debug.print("-", .{});
    std.debug.print("\n", .{});

    var single_rate: ?f64 = null;
    var batch_rate: ?f64 = null;

    if (opts.reverse) {
        if (opts.mode != .single) batch_rate = try timeOne(allocator, e, docs, byte_n, .batch);
        if (opts.mode != .batch) single_rate = try timeOne(allocator, e, docs, byte_n, .single);
    } else {
        if (opts.mode != .batch) single_rate = try timeOne(allocator, e, docs, byte_n, .single);
        if (opts.mode != .single) batch_rate = try timeOne(allocator, e, docs, byte_n, .batch);
    }

    if (single_rate != null and batch_rate != null) {
        std.debug.print(
            "\nbatch / single: {d:.3}x\n",
            .{batch_rate.? / single_rate.?},
        );
    }
}

/// Embeds every document both ways and reports how far apart the two answers are.
fn verify(allocator: std.mem.Allocator, e: *embed.Embedder, docs: []const Doc) !void {
    var worst_abs: f32 = 0;
    var worst_cos: f64 = 1;
    var worst_str: []const u8 = "";
    var compared: usize = 0;
    var null_mismatch: usize = 0;

    for (docs) |doc| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();

        const batched = try e.embedBatch(a, doc.sentences);
        for (doc.sentences, batched) |str, batch_out| {
            const single_out = try e.embed(a, str);
            if ((single_out == null) != (batch_out == null)) {
                null_mismatch += 1;
                continue;
            }
            if (single_out == null) continue;

            const x = single_out.?.slice();
            const y = batch_out.?.slice();
            var dot: f64 = 0;
            var max_abs: f32 = 0;
            for (x, y) |xi, yi| {
                dot += @as(f64, xi) * @as(f64, yi);
                max_abs = @max(max_abs, @abs(xi - yi));
            }
            compared += 1;
            if (dot < worst_cos) {
                worst_cos = dot;
                worst_str = str;
            }
            worst_abs = @max(worst_abs, max_abs);
        }
    }

    std.debug.print(
        \\compared        {d} vectors
        \\null mismatches {d}
        \\max |delta|     {e:.3}
        \\min cosine      {d:.9}
        \\worst string    '{s}'
        \\
    , .{
        compared,
        null_mismatch,
        worst_abs,
        worst_cos,
        worst_str[0..@min(worst_str.len, 60)],
    });
}

const Shape = enum { single, batch };

/// Embeds every document once and returns the achieved sentences/second.
fn timeOne(
    allocator: std.mem.Allocator,
    e: *embed.Embedder,
    docs: []const Doc,
    byte_n: usize,
    shape: Shape,
) !f64 {
    var sentence_n: usize = 0;
    var vec_checksum: f64 = 0;

    var timer = try std.time.Timer.start();
    for (docs) |doc| {
        // One arena per document, mirroring embedTextInternal: the embedder's output
        // lives exactly as long as the document being ingested.
        var doc_arena = std.heap.ArenaAllocator.init(allocator);
        defer doc_arena.deinit();
        const a = doc_arena.allocator();

        switch (shape) {
            .single => for (doc.sentences) |s| {
                if (try e.embed(a, s)) |v| vec_checksum += v.slice()[0];
                sentence_n += 1;
            },
            .batch => {
                const outs = try e.embedBatch(a, doc.sentences);
                for (outs) |maybe| if (maybe) |v| {
                    vec_checksum += v.slice()[0];
                };
                sentence_n += doc.sentences.len;
            },
        }
    }
    const ns = timer.read();

    const secs = @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
    const sents = @as(f64, @floatFromInt(sentence_n));
    // Keeps the checksum from being optimized away, and is a cheap tripwire: the two
    // shapes must agree, because embedBatch is defined as embed run per string.
    std.mem.doNotOptimizeAway(&vec_checksum);

    std.debug.print("{s: <7} {d: >5} {d: >11} {d: >12.3} {d: >9.1} {d: >8.1} {d: >10.3}\n", .{
        @tagName(shape),
        docs.len,
        sentence_n,
        secs,
        sents / secs,
        (@as(f64, @floatFromInt(byte_n)) / 1024.0) / secs,
        (secs * 1000.0) / sents,
    });

    return sents / secs;
}

/// Pays the lazy model load and whatever first-call cost the backend has, so it does not
/// land inside the first timed mode and make it look slow.
fn warmup(
    allocator: std.mem.Allocator,
    e: *embed.Embedder,
    docs: []const Doc,
    n: usize,
) !void {
    if (n == 0) return;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var done: usize = 0;
    var timer = try std.time.Timer.start();
    outer: for (docs) |doc| {
        for (doc.sentences) |s| {
            _ = try e.embed(arena.allocator(), s);
            done += 1;
            if (done >= n) break :outer;
        }
    }
    std.debug.print(
        "warmup     {d} sentences in {d:.3}s (model load included)\n\n",
        .{ done, @as(f64, @floatFromInt(timer.read())) / std.time.ns_per_s },
    );
}

/// One sentence per line. Newlines are sentence delimiters upstream, so no sentence can
/// contain one and the line count is exactly the sentence count.
fn dumpSentences(path: []const u8, docs: []const Doc) !void {
    var file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    var buf: [64 * 1024]u8 = undefined;
    var w = file.writer(&buf);
    for (docs) |doc| {
        for (doc.sentences) |s| {
            try w.interface.writeAll(s);
            try w.interface.writeByte('\n');
        }
    }
    try w.interface.flush();
}

// ***************************************************************************************** Corpus

fn loadDocs(arena: std.mem.Allocator, opts: Options) ![]Doc {
    var dir = std.fs.cwd().openDir(opts.corpus, .{ .iterate = true }) catch |err| {
        fatal("cannot open corpus dir '{s}': {t}", .{ opts.corpus, err });
    };
    defer dir.close();

    // The corpus directory holds hundreds of thousands of files; stop walking once there
    // are enough candidates to sample from rather than listing the whole thing.
    const pool_target = @max(opts.docs * 4, 4096);
    var names: std.ArrayList([]const u8) = .{};
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        try names.append(arena, try arena.dupe(u8, entry.name));
        if (names.items.len >= pool_target) break;
    }

    var prng = std.Random.DefaultPrng.init(opts.seed);
    prng.random().shuffle([]const u8, names.items);

    var docs: std.ArrayList(Doc) = .{};
    for (names.items) |name| {
        if (docs.items.len >= opts.docs) break;

        const contents = dir.readFileAlloc(arena, name, MAX_ARTICLE_BYTES) catch continue;

        var sentences: std.ArrayList([]const u8) = .{};
        var spliterator = embed.SentenceSpliterator.init(contents);
        while (spliterator.next()) |sentence| {
            if (whitespaceOnly(sentence.contents) or !wordlike(sentence.contents)) continue;
            try sentences.append(arena, sentence.contents);
            if (opts.max_sentences != 0 and sentences.items.len >= opts.max_sentences) break;
        }
        if (sentences.items.len == 0) continue;

        try docs.append(arena, .{ .name = name, .sentences = sentences.items });
    }
    return docs.items;
}

/// Copied from vector.zig, which keeps both private. Duplicated rather than exported
/// because the benchmark must not be able to change what the engine filters.
fn whitespaceOnly(contents: []const u8) bool {
    for (contents) |c| {
        if (!std.ascii.isWhitespace(c)) return false;
    }
    return true;
}

fn wordlike(contents: []const u8) bool {
    var n_alphanumeral: usize = 0;
    for (contents) |c| {
        if (std.ascii.isAlphanumeric(c)) {
            n_alphanumeral += 1;
            if (n_alphanumeral > 1) return true;
        }
    }
    return false;
}

// ******************************************************************************************* Args

fn parseArgs(args: []const []const u8) !Options {
    var opts = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--corpus")) {
            opts.corpus = nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--model")) {
            opts.model = std.meta.stringToEnum(Model, nextArg(args, &i)) orelse
                fatal("--model must be mpnet, llama or nl", .{});
        } else if (std.mem.eql(u8, arg, "--mode")) {
            opts.mode = std.meta.stringToEnum(Mode, nextArg(args, &i)) orelse
                fatal("--mode must be single, batch or both", .{});
        } else if (std.mem.eql(u8, arg, "--docs")) {
            opts.docs = try std.fmt.parseInt(usize, nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            opts.seed = try std.fmt.parseInt(u64, nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            opts.warmup = try std.fmt.parseInt(usize, nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--verify")) {
            opts.verify = true;
        } else if (std.mem.eql(u8, arg, "--reverse")) {
            opts.reverse = true;
        } else if (std.mem.eql(u8, arg, "--dump")) {
            opts.dump = nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--max-sentences")) {
            opts.max_sentences = try std.fmt.parseInt(usize, nextArg(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            usage(0);
        } else {
            fatal("unknown argument '{s}'", .{arg});
        }
    }
    return opts;
}

fn nextArg(args: []const []const u8, i: *usize) []const u8 {
    i.* += 1;
    if (i.* >= args.len) fatal("'{s}' needs a value", .{args[i.* - 1]});
    return args[i.*];
}

fn usage(code: u8) noreturn {
    std.debug.print(
        \\usage: embedbench [options]
        \\
        \\  --corpus <dir>     article directory (default: wikitest/wikidata/md)
        \\  --model <m>        mpnet | llama | nl (default: mpnet)
        \\  --mode <m>         single | batch | both (default: both)
        \\  --docs <n>         documents to embed (default: 200)
        \\  --seed <n>         article sampling seed (default: 1)
        \\  --warmup <n>       sentences to embed untimed first (default: 32)
        \\  --max-sentences <n>  cap sentences per document (default: uncapped)
        \\  --verify           compare batch against single instead of timing
        \\  --reverse          time batch before single
        \\  --dump <file>      write the selected sentences one per line and exit
        \\
    , .{});
    std.process.exit(code);
}

fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
