//! dve against BEIR, the standard zero-shot retrieval benchmark.
//!
//! Everything else we measure is self-referential. `--verify` in wikitest scores the index
//! against an exhaustive exact-cosine scan of our own store, which answers "did stage one
//! surface what stage two would have ranked first" and nothing about whether that answer is
//! any good -- "capital city of france" scored a perfect 1.000 there while returning a
//! baseball league. `src/benchmark.zig` scores the embedder on corpora of 2 to 14 documents,
//! which is too small for a wrong answer to have anywhere to come from.
//!
//! BEIR has human relevance judgments and published per-model numbers. That buys the one
//! measurement we cannot make ourselves: run dve on the same dataset as a published score for
//! the same embedding model, and the gap between them is *ours* -- our chunking, our
//! thresholds, our ranking. Without it we cannot tell a mediocre embedder from a good
//! embedder we are damaging.
//!
//! Metrics are the usual three. nDCG@10 is the headline BEIR reports; Recall@100 says whether
//! the answer was anywhere in reach; MRR@10 says how far down it was.
//!
//!   experiments/beirbench/download.sh
//!   zig build beirbench -Dllama -Doptimize=ReleaseFast
//!   ./zig-out/bin/beirbench --model llama

const std = @import("std");
const dve = @import("dve");
const Timer = dve.util.Timer;
const nanos = dve.util.nanos;

pub const std_options: std.Options = .{ .log_level = .warn };

const MAX_JSONL_BYTES: usize = 512 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 1024 * 1024;

const Model = enum {
    mpnet,
    nl,
    llama,

    fn id(self: Model) dve.embed.EmbeddingModel {
        return switch (self) {
            .mpnet => .mpnet_embedding,
            .nl => .apple_nlembedding,
            .llama => .llama_nomic_embed_text_v1_5_f32,
        };
    }
};

const Options = struct {
    dataset: []const u8 = "experiments/beirbench/data/scifact",
    db: []const u8 = "experiments/beirbench/data/beirbench-db",
    model: Model = .llama,
    /// Which qrels split to score against. BEIR's published numbers are on test.
    split: []const u8 = "test",
    /// nDCG and MRR cut-off. 10 is what BEIR reports.
    k: usize = 10,
    recall_at: usize = 100,
    /// Stage one's candidate budget. Higher than the engine default because this collapses
    /// chunk hits to one per document and still wants 100 distinct documents out the far end;
    /// 500 chunk candidates over a corpus with several chunks per document does not reliably
    /// yield that many.
    candidates: usize = 2000,
    /// Similarity floor. Zero by default: nDCG and recall are ranking metrics, and dropping
    /// results under an absolute threshold truncates the ranking and makes the number
    /// incomparable with anyone else's. Pass the model's real threshold to see what a user
    /// would actually get.
    threshold: f32 = 0,
    /// Cap on judged queries, for a quick smoke run.
    limit: usize = 0,
    /// Skip ingest if the database already holds vectors.
    reuse: bool = false,
    progress: usize = 500,
    /// Print this many of the worst-scoring queries, to see what the misses look like.
    show_failures: usize = 0,
};

const Doc = struct { id: []const u8, title: []const u8, text: []const u8 };
const Query = struct { id: []const u8, text: []const u8 };

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const opts = try parseArgs(args);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const corpus = try loadCorpus(a, io, opts.dataset);
    const queries = try loadQueries(a, io, opts.dataset);
    var qrels = try loadQrels(a, io, opts.dataset, opts.split);

    std.debug.print(
        \\dataset    {s}
        \\corpus     {d} documents
        \\queries    {d} total, {d} judged in '{s}'
        \\model      {t}
        \\settings   k={d} recall@{d} candidates={d} threshold={d:.2}
        \\
        \\
    , .{
        opts.dataset,
        corpus.len,
        queries.count(),
        qrels.count(),
        opts.split,
        opts.model,
        opts.k,
        opts.recall_at,
        opts.candidates,
        opts.threshold,
    });

    switch (opts.model) {
        inline else => |m| try run(m.id(), allocator, io, opts, corpus, queries, &qrels),
    }
}

fn run(
    comptime model: dve.embed.EmbeddingModel,
    allocator: std.mem.Allocator,
    io: std.Io,
    opts: Options,
    corpus: []const Doc,
    queries: std.StringHashMap([]const u8),
    qrels: *std.StringHashMap(std.ArrayList([]const u8)),
) !void {
    const Engine = dve.VectorEngine(model);

    var db_dir = try std.Io.Dir.cwd().createDirPathOpen(io, opts.db, .{ .open_options = .{ .iterate = true } });
    defer db_dir.close(io);

    var load_timer = Timer.start(io);
    const db = try Engine.init(allocator, io, db_dir, .{ .candidates = opts.candidates });
    defer db.deinit();
    std.debug.print("opened database in {f}\n", .{nanos(load_timer.read())});
    db.embedder.threshold = opts.threshold;

    if (db.vec_storage.vec_n != 0 and opts.reuse) {
        std.debug.print("reusing {d} vectors already indexed\n\n", .{db.vec_storage.vec_n});
    } else {
        if (db.vec_storage.vec_n != 0) std.debug.print(
            "note: database holds {d} vectors; appending (pass --reuse to skip ingest)\n",
            .{db.vec_storage.vec_n},
        );
        try ingest(Engine, db, allocator, io, opts, corpus);
    }

    // *** score ***
    var sum_ndcg: f64 = 0;
    var sum_recall: f64 = 0;
    var sum_mrr: f64 = 0;
    var scored_n: usize = 0;

    const buf = try allocator.alloc(dve.SearchResult, opts.recall_at);
    defer allocator.free(buf);

    const Failure = struct { id: []const u8, ndcg: f64, rank: ?usize, top: []const u8 };
    var failures: std.ArrayList(Failure) = .empty;
    defer failures.deinit(allocator);

    var search_ns: u64 = 0;
    var timer = Timer.start(io);

    var it = qrels.iterator();
    while (it.next()) |entry| {
        if (opts.limit != 0 and scored_n >= opts.limit) break;
        const qid = entry.key_ptr.*;
        const relevant = entry.value_ptr.items;
        const qtext = queries.get(qid) orelse continue;

        timer.reset();
        const n = try db.uniqueSearch(qtext, buf);
        search_ns += timer.read();

        const m = score(buf[0..n], relevant, opts.k);
        sum_ndcg += m.ndcg;
        sum_recall += m.recall;
        sum_mrr += m.mrr;
        scored_n += 1;

        if (opts.show_failures != 0 and m.ndcg < 1.0) try failures.append(allocator, .{
            .id = qid,
            .ndcg = m.ndcg,
            .rank = m.first_rank,
            .top = if (n > 0) buf[0].path else "<nothing>",
        });
    }

    if (scored_n == 0) fatal("no judged queries were scored", .{});
    const d: f64 = @floatFromInt(scored_n);

    std.debug.print(
        \\
        \\=================== scoreboard ===================
        \\queries scored  {d}
        \\nDCG@{d}         {d:.4}
        \\Recall@{d}      {d:.4}
        \\MRR@{d}          {d:.4}
        \\search          {d:.2} ms/query
        \\==================================================
        \\
    , .{
        scored_n,
        opts.k,
        sum_ndcg / d,
        opts.recall_at,
        sum_recall / d,
        opts.k,
        sum_mrr / d,
        (@as(f64, @floatFromInt(search_ns)) / @as(f64, @floatFromInt(scored_n))) / std.time.ns_per_ms,
    });

    if (opts.show_failures != 0 and failures.items.len != 0) {
        std.sort.block(Failure, failures.items, {}, struct {
            fn lessThan(_: void, x: Failure, y: Failure) bool {
                return x.ndcg < y.ndcg;
            }
        }.lessThan);
        std.debug.print("\nworst {d} of {d} imperfect queries:\n", .{
            @min(opts.show_failures, failures.items.len),
            failures.items.len,
        });
        for (failures.items[0..@min(opts.show_failures, failures.items.len)]) |f| {
            const qtext = queries.get(f.id) orelse "";
            std.debug.print("  nDCG {d:.3} ", .{f.ndcg});
            if (f.rank) |r| {
                std.debug.print("(gold at rank {d}) ", .{r});
            } else {
                std.debug.print("(gold not in top {d}) ", .{opts.recall_at});
            }
            std.debug.print("q{s}: '{s}'\n    top hit: {s}\n", .{
                f.id,
                qtext[0..@min(qtext.len, 70)],
                f.top,
            });
        }
    }
}

fn ingest(
    comptime Engine: type,
    db: *Engine,
    allocator: std.mem.Allocator,
    io: std.Io,
    opts: Options,
    corpus: []const Doc,
) !void {
    std.debug.print("\ningesting {d} documents...\n", .{corpus.len});
    var timer = Timer.start(io);

    for (corpus, 1..) |doc, i| {
        // Title on its own line, which is how a note is actually shaped and which makes the
        // title its own chunk -- titles are high signal and short, so this is the natural
        // form rather than a trick. It does mean the chunker's newline behaviour is part of
        // what this benchmark measures, which is the point.
        const body = if (doc.title.len != 0)
            try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ doc.title, doc.text })
        else
            try allocator.dupe(u8, doc.text);
        defer allocator.free(body);

        while (true) {
            db.embedTextAsync(doc.id, body) catch |err| switch (err) {
                error.Full => {
                    io.sleep(.fromNanoseconds(250 * std.time.ns_per_us), .awake) catch {};
                    continue;
                },
                else => {
                    std.debug.print("skip {s}: {t}\n", .{ doc.id, err });
                    break;
                },
            };
            break;
        }

        if (i % opts.progress == 0) std.debug.print(
            "  {d}/{d} submitted, {d} vectors, {f}\n",
            .{ i, corpus.len, db.vec_storage.vec_n, nanos(timer.read()) },
        );
    }

    db.shutdown();
    std.debug.print(
        "ingested {d} documents -> {d} vectors in {f}\n",
        .{ corpus.len, db.vec_storage.vec_n, nanos(timer.read()) },
    );
}

// ***************************************************************************************** Metrics

const Metrics = struct {
    ndcg: f64,
    recall: f64,
    mrr: f64,
    /// 1-based rank of the first relevant document, if one was retrieved at all.
    first_rank: ?usize,
};

/// `results` is ranked best-first and already one entry per document.
///
/// SciFact's judgments are binary, so exponential gain (2^rel - 1) and linear gain agree; the
/// exponential form is written out because it is what TREC and BEIR use and a graded dataset
/// would need it.
fn score(results: []const dve.SearchResult, relevant: []const []const u8, k: usize) Metrics {
    var dcg: f64 = 0;
    var hits_at_k: usize = 0;
    var hits_all: usize = 0;
    var first_rank: ?usize = null;

    for (results, 1..) |r, rank| {
        if (!contains(relevant, r.path)) continue;
        hits_all += 1;
        if (first_rank == null) first_rank = rank;
        if (rank <= k) {
            hits_at_k += 1;
            dcg += 1.0 / std.math.log2(@as(f64, @floatFromInt(rank + 1)));
        }
    }

    // Ideal ranking: every relevant document packed into the first positions.
    var idcg: f64 = 0;
    for (1..@min(k, relevant.len) + 1) |rank| {
        idcg += 1.0 / std.math.log2(@as(f64, @floatFromInt(rank + 1)));
    }

    return .{
        .ndcg = if (idcg > 0) dcg / idcg else 0,
        .recall = if (relevant.len > 0)
            @as(f64, @floatFromInt(hits_all)) / @as(f64, @floatFromInt(relevant.len))
        else
            0,
        .mrr = if (first_rank) |r| (if (r <= k) 1.0 / @as(f64, @floatFromInt(r)) else 0) else 0,
        .first_rank = first_rank,
    };
}

fn contains(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |h| if (std.mem.eql(u8, h, needle)) return true;
    return false;
}

// ****************************************************************************************** Corpus

fn loadCorpus(arena: std.mem.Allocator, io: std.Io, dataset: []const u8) ![]Doc {
    const path = try std.Io.Dir.path.join(arena, &.{ dataset, "corpus.jsonl" });
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(MAX_JSONL_BYTES)) catch |e| {
        fatal("cannot read '{s}': {t}\n(run experiments/beirbench/download.sh first)", .{ path, e });
    };

    const Row = struct { _id: []const u8, title: []const u8 = "", text: []const u8 = "" };
    var out: std.ArrayList(Doc) = .empty;
    var lines = std.mem.tokenizeScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const parsed = std.json.parseFromSlice(Row, arena, line, .{
            .ignore_unknown_fields = true,
        }) catch continue;
        try out.append(arena, .{
            .id = parsed.value._id,
            .title = parsed.value.title,
            .text = parsed.value.text,
        });
    }
    return out.items;
}

fn loadQueries(arena: std.mem.Allocator, io: std.Io, dataset: []const u8) !std.StringHashMap([]const u8) {
    const path = try std.Io.Dir.path.join(arena, &.{ dataset, "queries.jsonl" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(MAX_JSONL_BYTES));

    const Row = struct { _id: []const u8, text: []const u8 = "" };
    var out = std.StringHashMap([]const u8).init(arena);
    var lines = std.mem.tokenizeScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const parsed = std.json.parseFromSlice(Row, arena, line, .{
            .ignore_unknown_fields = true,
        }) catch continue;
        try out.put(parsed.value._id, parsed.value.text);
    }
    return out;
}

/// `query-id<TAB>corpus-id<TAB>score`, with a header line. Only positive scores are kept: a
/// zero in a qrels file means "judged and not relevant", which is not the same as unjudged
/// but counts the same way here.
fn loadQrels(
    arena: std.mem.Allocator,
    io: std.Io,
    dataset: []const u8,
    split: []const u8,
) !std.StringHashMap(std.ArrayList([]const u8)) {
    const name = try std.fmt.allocPrint(arena, "{s}.tsv", .{split});
    const path = try std.Io.Dir.path.join(arena, &.{ dataset, "qrels", name });
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(MAX_JSONL_BYTES)) catch |e| {
        fatal("cannot read '{s}': {t}", .{ path, e });
    };

    var out = std.StringHashMap(std.ArrayList([]const u8)).init(arena);
    var lines = std.mem.tokenizeScalar(u8, bytes, '\n');
    _ = lines.next(); // header
    while (lines.next()) |line| {
        var cols = std.mem.tokenizeAny(u8, line, "\t\r");
        const qid = cols.next() orelse continue;
        const did = cols.next() orelse continue;
        const rel = std.fmt.parseInt(i32, cols.next() orelse "0", 10) catch 0;
        if (rel <= 0) continue;

        const gop = try out.getOrPut(qid);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(arena, did);
    }
    return out;
}

// ******************************************************************************************** Args

fn parseArgs(args: []const []const u8) !Options {
    var opts = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--dataset")) {
            opts.dataset = next(args, &i);
        } else if (std.mem.eql(u8, arg, "--db")) {
            opts.db = next(args, &i);
        } else if (std.mem.eql(u8, arg, "--model")) {
            opts.model = std.meta.stringToEnum(Model, next(args, &i)) orelse
                fatal("--model must be mpnet, nl or llama", .{});
        } else if (std.mem.eql(u8, arg, "--split")) {
            opts.split = next(args, &i);
        } else if (std.mem.eql(u8, arg, "--k")) {
            opts.k = try std.fmt.parseInt(usize, next(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--recall-at")) {
            opts.recall_at = try std.fmt.parseInt(usize, next(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--candidates")) {
            opts.candidates = try std.fmt.parseInt(usize, next(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--threshold")) {
            opts.threshold = try std.fmt.parseFloat(f32, next(args, &i));
        } else if (std.mem.eql(u8, arg, "--limit")) {
            opts.limit = try std.fmt.parseInt(usize, next(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--reuse")) {
            opts.reuse = true;
        } else if (std.mem.eql(u8, arg, "--progress")) {
            opts.progress = try std.fmt.parseInt(usize, next(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "--show-failures")) {
            opts.show_failures = try std.fmt.parseInt(usize, next(args, &i), 10);
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            usage(0);
        } else {
            fatal("unknown argument '{s}'", .{arg});
        }
    }
    return opts;
}

fn next(args: []const []const u8, i: *usize) []const u8 {
    i.* += 1;
    if (i.* >= args.len) fatal("'{s}' needs a value", .{args[i.* - 1]});
    return args[i.*];
}

fn usage(code: u8) noreturn {
    std.debug.print(
        \\usage: beirbench [options]
        \\
        \\  --dataset <dir>     BEIR dataset directory (default: experiments/beirbench/data/scifact)
        \\  --db <dir>          index directory
        \\  --model <m>         mpnet | nl | llama      (default: llama)
        \\  --split <name>      qrels split to score    (default: test)
        \\  --k <n>             nDCG/MRR cut-off        (default: 10)
        \\  --recall-at <n>     recall cut-off          (default: 100)
        \\  --candidates <n>    stage-one budget        (default: 2000)
        \\  --threshold <f>     similarity floor        (default: 0)
        \\  --limit <n>         cap judged queries
        \\  --reuse             skip ingest if the index is populated
        \\  --show-failures <n> print the n worst queries
        \\
    , .{});
    std.process.exit(code);
}

fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
