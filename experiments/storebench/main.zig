//! A/B benchmark: `vec_storage.zig` (v1, RAM-resident) vs `vstore.zig` (v2, disk-resident).
//!
//! Both stores are driven directly, never through `vector.zig`. Embedding costs ~20 ms per
//! sentence and would bury a storage difference measured in microseconds, so the corpus here
//! is synthetic -- clustered unit vectors, which is what the real embedder produces and what
//! makes `search`'s threshold do work rather than reject everything.
//!
//! The comparison is not symmetric, and pretending otherwise would flatter one side:
//!
//!   * v1's `put` is a memory write. Nothing reaches disk until `save`, which serializes the
//!     *entire* capacity -- every vector, live or not -- and returns once the bytes are in the
//!     page cache.
//!   * v2's `put` is two `pwrite`s and is durable to the page cache on return. `flush` is an
//!     `F_FULLFSYNC` barrier, which waits for the drive to commit its own write cache.
//!
//! So "v1 save vs v2 flush" compares *written* against *committed*. Both are reported, plus
//! v2 with no barrier at all, so the gap can be read either way.
//!
//! Phases run independently (`--phase`) so an external script can `sudo purge` between them
//! and measure the cold regime, which is the one that matters at filesystem scale.

const PHASES = "ingest, search, open, all";

const Phase = enum { ingest, search, open, all };
const Which = enum {
    /// vec_storage.zig: every vector resident as f32, search is a linear scan over RAM.
    v1,
    /// vstore.zig alone: every vector on disk, search is a linear scan over the file.
    v2,
    /// vstore.zig plus codes.zig: scan 384-bit codes in RAM for K candidates, then read those
    /// K vectors from disk and rank them by exact cosine. The stack as it actually ships.
    v2_indexed,
    /// v1 and v2_indexed -- the old engine against the new one, which is the comparison that
    /// matters now that vector.zig only has the one.
    both,
    all,
};

const Options = struct {
    n: usize = 100_000,
    /// Sentences per document. Each document is one put-batch followed by one persist call,
    /// which is exactly what `embedText` does, so this is the unit the real workload has.
    doc: usize = 20,
    queries: usize = 200,
    /// Candidates stage one hands stage two, for the indexed run. Also the number of vectors
    /// it reads from disk per query.
    k: usize = 500,
    /// Persist once every this many documents. 1 is what `embedText` does. It is also what
    /// makes v1's ingest quadratic -- `save` rewrites the entire capacity every call, so the
    /// total written is documents x capacity -- which is measurable at 20k vectors and takes
    /// hours at 100k. Raising it amortizes that; `--no-persist` removes it.
    persist_every: usize = 1,
    phase: Phase = .all,
    which: Which = .both,
    /// Skip the persist call entirely, leaving v1's puts in RAM and v2's in the page cache.
    /// Isolates the put path from the durability path.
    no_persist: bool = false,
    dir: []const u8 = "bench-data",
    keep: bool = false,
    /// Skip ingest and measure the databases already in `--dir`. This is what makes a cold
    /// measurement possible: ingest in one process, `sudo purge`, then reopen in another.
    /// Without it every phase rebuilds the corpus first and warms the page cache doing so.
    reuse: bool = false,
    seed: u64 = 42,
    threshold: f32 = 0.5,
};

// ************************************************************************************ Corpus
/// Random unit vectors in 768 dimensions are very nearly orthogonal, so a 0.5 threshold would
/// reject every one of them and `search` would never touch its priority queue. Real embeddings
/// cluster; this draws from CENTROIDS of them so a realistic slice of the corpus clears the
/// threshold on every query.
const CENTROIDS = 64;
/// Noise relative to the centroid. Picked so a vector's similarity to its own centroid lands
/// near 0.7 and to any other near 0, which is roughly what mpnet output looks like.
const SPREAD: f32 = 1.0;

fn randomUnit(comptime N: usize, rng: std.Random, out: *[N]f32) void {
    var sum: f32 = 0;
    for (out) |*c| {
        c.* = rng.floatNorm(f32);
        sum += c.* * c.*;
    }
    const inv = 1.0 / @sqrt(sum);
    for (out) |*c| c.* *= inv;
}

const Corpus = struct {
    centroids: [][VEC_SZ]f32,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, rng: std.Random) !Corpus {
        const cs = try allocator.alloc([VEC_SZ]f32, CENTROIDS);
        for (cs) |*c| randomUnit(VEC_SZ, rng, c);
        return .{ .centroids = cs, .allocator = allocator };
    }
    fn deinit(self: *Corpus) void {
        self.allocator.free(self.centroids);
    }

    /// A unit vector near centroid `i % CENTROIDS`. Deterministic in `rng`, so v1 and v2 are
    /// handed byte-identical corpora and any difference in results is the store's.
    fn draw(self: *const Corpus, rng: std.Random, i: usize, out: *[VEC_SZ]f32) void {
        const c = &self.centroids[i % CENTROIDS];
        var sum: f32 = 0;
        for (out, c) |*o, cv| {
            o.* = cv + SPREAD * rng.floatNorm(f32) / @sqrt(@as(f32, VEC_SZ));
            sum += o.* * o.*;
        }
        const inv = 1.0 / @sqrt(sum);
        for (out) |*o| o.* *= inv;
    }
};

// ************************************************************************************ Results
const Result = struct {
    label: []const u8,
    /// Time in the put calls alone, with every persist call excluded.
    put_ns: u64 = 0,
    /// Time in `save`/`flush`, summed across documents.
    persist_ns: u64 = 0,
    open_ns: u64 = 0,
    persists: u64 = 0,
    /// Time to build the code index by reading every vector, and its resident size. Zero for
    /// an unindexed run.
    index_ns: u64 = 0,
    index_bytes: u64 = 0,
    /// Writing that index to a file and reading it back. The point of the file is that the
    /// second number replaces the first at every open.
    index_save_ns: u64 = 0,
    index_load_ns: u64 = 0,
    search_ns: u64 = 0,
    hits: u64 = 0,
    bytes: u64 = 0,
    /// Absolute peak RSS, not a delta. `maxrss` is a high-water mark for the whole process,
    /// so the second store to run in a `--store both` pass inherits the first one's peak and
    /// a delta would read as zero. Only a single-store run reports this meaningfully.
    rss_peak: u64 = 0,
    vec_n: usize = 0,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const opts = parseArgs(args);

    std.Io.Dir.cwd().createDirPath(io, opts.dir) catch {};
    var dir = try std.Io.Dir.cwd().openDir(io, opts.dir, .{ .iterate = true });
    defer dir.close(io);
    defer if (!opts.keep) std.Io.Dir.cwd().deleteTree(io, opts.dir) catch {};

    std.debug.print(
        \\storebench -- vec_storage.zig (v1) vs vstore.zig (v2)
        \\
        \\vectors    {d} x {d}-dim f32 ({f} of payload){s}
        \\documents  {d} of {d} sentences
        \\queries    {d} at threshold {d:.2}
        \\persist    {s}
        \\dir        {s}
        \\
        \\
    , .{
        opts.n,
        VEC_SZ,
        fmtBytes(opts.n * VEC_SZ * 4),
        if (opts.reuse) " -- ingest skipped, reusing --dir" else "",
        (opts.n + opts.doc - 1) / opts.doc,
        opts.doc,
        opts.queries,
        opts.threshold,
        persistLabel(opts),
        opts.dir,
    });

    var results: std.ArrayList(Result) = .empty;
    defer results.deinit(gpa);

    const want = opts.which;
    if (want == .v1 or want == .both or want == .all) {
        try results.append(gpa, try runOne(gpa, io, dir, opts, V1, false, "v1 vec_storage"));
    }
    if (want == .v2 or want == .all) {
        try results.append(gpa, try runOne(gpa, io, dir, opts, V2, false, "v2 vstore"));
    }
    if (want == .v2_indexed or want == .both or want == .all) {
        try results.append(gpa, try runOne(gpa, io, dir, opts, V2, true, "v2 vstore+codes"));
    }

    report(results.items, opts);
}

/// One store, all phases. Generic over the two store types rather than duplicated, but the
/// call shapes differ enough that each `if (Store == V1)` below is a real API difference and
/// not an abstraction waiting to be factored out.
fn runOne(
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    opts: Options,
    comptime Store: type,
    /// Search through the codes index rather than scanning the store. Only meaningful for V2;
    /// v1 has no index and never will.
    indexed: bool,
    label: []const u8,
) !Result {
    var res = Result{ .label = label };
    const path = if (Store == V1) "bench-v1.db" else "bench-v2.db";
    if (!opts.reuse) dir.deleteFile(io, path) catch {};

    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rng = prng.random();
    var corpus = try Corpus.init(gpa, rng);
    defer corpus.deinit();

    // ---------------------------------------------------------------------------- ingest
    if (!opts.reuse) {
        // v1 preallocates and doubles; v2 grows a chunk at a time. Both start where
        // `vector.zig` starts them, so the growth cost each design carries is in the number.
        var store = if (Store == V1)
            try Store.init(gpa, io, dir, .{})
        else
            try Store.init(gpa, io, dir, .{ .path = path });
        defer store.deinit();

        var vec: [VEC_SZ]f32 = undefined;
        var put_timer = Timer.start(io);
        var persist_ns: u64 = 0;
        var put_ns: u64 = 0;

        var i: usize = 0;
        var doc: usize = 1;
        while (i < opts.n) : (doc += 1) {
            const end = @min(i + opts.doc, opts.n);
            put_timer.reset();
            while (i < end) : (i += 1) {
                corpus.draw(rng, i, &vec);
                if (Store == V1) {
                    _ = try store.put(@intCast(doc), i, i + 1, vec);
                } else {
                    _ = try store.put(.{ .doc_id = @intCast(doc), .start_i = i, .end_i = i + 1 }, &vec);
                }
            }
            put_ns += put_timer.read();

            if (opts.no_persist) continue;
            if (doc % opts.persist_every != 0 and i < opts.n) continue;
            put_timer.reset();
            if (Store == V1) try store.save(path) else try store.flush();
            persist_ns += put_timer.read();
            res.persists += 1;
        }
        res.put_ns = put_ns;
        res.persist_ns = persist_ns;
        res.vec_n = store.vec_n;

        // v1's puts live in RAM until something calls save, so an unpersisted run has nothing
        // on disk to reopen. One save at the end gives the open and search phases a file.
        if (opts.no_persist and Store == V1) try store.save(path);
    }

    res.rss_peak = rssBytes();
    res.bytes = (dir.statFile(io, path, .{}) catch |e| switch (e) {
        error.FileNotFound => fatal("{s}: no database at {s}/{s} to reuse", .{ label, opts.dir, path }),
        else => return e,
    }).size;

    if (opts.phase == .ingest) return res;

    // ------------------------------------------------------------------------------ open
    // v1 reads the whole file into its arrays; v2 scans every metadata trailer. Both are O(n),
    // and this is the number the sidecar index exists to remove.
    var open_timer = Timer.start(io);
    var store = if (Store == V1)
        try Store.init(gpa, io, dir, .{})
    else
        try Store.init(gpa, io, dir, .{ .path = path });
    defer store.deinit();
    if (Store == V1) try store.load(path);
    res.open_ns = open_timer.read();
    // In a reuse run the ingest phase never ran, so the live count comes from the file.
    if (res.vec_n == 0) res.vec_n = store.vec_n;

    if (opts.phase == .open) return res;

    // ------------------------------------------------------------------------ build index
    var idx: ?Codes = null;
    defer if (idx) |*c| c.deinit();
    if (Store == V2 and indexed) {
        var c = try Codes.init(gpa, io, .{ .capacity = store.slot_n });
        errdefer c.deinit();
        var build_timer = Timer.start(io);
        var it = try store.iterate();
        defer it.deinit();
        while (try it.next()) |e| try c.put(e.slot, e.vec);
        res.index_ns = build_timer.read();
        res.index_bytes = c.bytes();

        // Round-trip it, so the build cost and the load cost are measured side by side on the
        // same index. `load` consumes the file, which is how a saved index is kept from
        // outliving the process that wrote it, so this saves again afterwards.
        const stamp = codesStamp(&store);
        var save_timer = Timer.start(io);
        try c.save(dir, CODES_FILE, stamp);
        res.index_save_ns = save_timer.read();

        var load_timer = Timer.start(io);
        var reloaded = (try Codes.load(gpa, io, dir, CODES_FILE, stamp, .{})) orelse
            fatal("the index just written would not load back", .{});
        res.index_load_ns = load_timer.read();

        c.deinit();
        idx = reloaded;
        // Re-sampled: the peak above was taken before the index existed, so an indexed run
        // would otherwise report the store's footprint and not its own.
        res.rss_peak = rssBytes();
        // Searches below run against the *loaded* index, so a save/load bug shows up as bad
        // results rather than only as a bad number.
        _ = &reloaded;
    }

    // ---------------------------------------------------------------------------- search
    const buf = try gpa.alloc(Store.SearchEntry, 50);
    defer gpa.free(buf);
    const cands = try gpa.alloc(Codes.Candidate, opts.k);
    defer gpa.free(cands);

    // Queries are drawn from the same centroids as the corpus, so every one of them has real
    // neighbours to find. A query with no hits would measure the scan and nothing else.
    var qprng = std.Random.DefaultPrng.init(opts.seed +% 1);
    const qrng = qprng.random();
    var query: [VEC_SZ]f32 = undefined;
    var search_timer = Timer.start(io);
    var search_ns: u64 = 0;
    for (0..opts.queries) |q| {
        corpus.draw(qrng, q, &query);
        search_timer.reset();
        const found = found: {
            if (Store == V2) {
                if (idx) |*c| {
                    break :found try twoStage(&store, c, &query, cands, buf, opts.threshold);
                }
                break :found try store.search(&query, buf, opts.threshold);
            }
            break :found try store.search(query, buf, opts.threshold);
        };
        search_ns += search_timer.read();
        res.hits += found;
    }
    res.search_ns = search_ns;
    return res;
}

/// The shipped search path, in miniature: codes for candidates, disk for the truth. Kept here
/// rather than called through `vector.zig` so the comparison measures storage and indexing
/// without an embedder in the way.
const CODES_FILE = "bench.codes";

fn codesStamp(store: *V2) dve.codes.Stamp {
    return .{
        .store_bytes = store.file.size() catch 0,
        .slot_n = store.slot_n,
        .vec_n = store.vec_n,
    };
}

fn twoStage(
    store: *V2,
    idx: *Codes,
    query: *const [VEC_SZ]f32,
    cands: []Codes.Candidate,
    out: []V2.SearchEntry,
    threshold: f32,
) !usize {
    const n_cand = try idx.search(query, cands);
    var vec: [VEC_SZ]f32 = undefined;
    var n: usize = 0;
    for (cands[0..n_cand]) |cand| {
        const row = (try store.getSlot(cand.slot, &vec)) orelse continue;
        const sim = storedDotAt(VEC_SZ, f32, &vec, query);
        if (sim <= threshold) continue;
        // Same bounded insertion the engine does: `out` is small, and most candidates lose to
        // the worst kept entry immediately.
        if (n == out.len and sim <= out[n - 1].similarity) continue;
        if (n < out.len) n += 1;
        var j = n - 1;
        while (j > 0 and out[j - 1].similarity < sim) : (j -= 1) out[j] = out[j - 1];
        out[j] = .{ .row = row, .similarity = sim };
    }
    return n;
}

// ************************************************************************************ Report
fn report(results: []const Result, opts: Options) void {
    const w = std.debug.print;
    w("{s:<18} {s:>12} {s:>12} {s:>12} {s:>12} {s:>12}\n", .{
        "", "put", "persist", "open", "search", "on disk",
    });
    w("{s:<18} {s:>12} {s:>12} {s:>12} {s:>12} {s:>12}\n", .{
        "", "vec/s", "total", "total", "ms/query", "",
    });
    w("{s:-<18} {s:->12} {s:->12} {s:->12} {s:->12} {s:->12}\n", .{ "", "", "", "", "", "" });

    for (results) |r| {
        const put_rate = rate(r.vec_n, r.put_ns);
        _ = r.persists;
        const ms_q = if (opts.queries == 0) 0 else msOf(r.search_ns) / @as(f64, @floatFromInt(opts.queries));
        w("{s:<18} {d:>12.0} {f} {f} {d:>12.3} {f:>12}\n", .{
            r.label,
            put_rate,
            Nanos{ .ns = r.persist_ns, .width = 12 },
            Nanos{ .ns = r.open_ns, .width = 12 },
            ms_q,
            fmtBytes(r.bytes),
        });
    }

    w("\n{s:<18} {s:>12} {s:>12} {s:>12} {s:>12}\n", .{
        "", "live vecs", "hits/query", "peak rss", "index size",
    });
    w("{s:-<18} {s:->12} {s:->12} {s:->12} {s:->12}\n", .{ "", "", "", "", "" });
    for (results) |r| {
        const hpq = if (opts.queries == 0) 0 else @as(f64, @floatFromInt(r.hits)) /
            @as(f64, @floatFromInt(opts.queries));
        w("{s:<18} {d:>12} {d:>12.1} {f:>12} {f:>12}\n", .{
            r.label, r.vec_n, hpq, fmtBytes(r.rss_peak), fmtBytes(r.index_bytes),
        });
    }

    for (results) |r| {
        if (r.index_ns == 0) continue;
        w("\n{s}: index build {f} (reads every vector), save {f}, load {f}\n", .{
            r.label, nanos(r.index_ns), nanos(r.index_save_ns), nanos(r.index_load_ns),
        });
        w("  loading instead of rebuilding is {d:.0}x, and that ratio is what grows:\n" ++
            "  the build reads the whole store, the load reads {f}.\n", .{
            @as(f64, @floatFromInt(r.index_ns)) /
                @as(f64, @floatFromInt(@max(r.index_load_ns, 1))),
            fmtBytes(r.index_bytes),
        });
    }
    if (results.len > 1) {
        w("\nrss is a process high-water mark, so it only separates the two stores when each\n" ++
            "is run on its own: --store v1 and --store v2 in separate processes.\n", .{});
    }

    if (results.len == 2) {
        const a = results[0];
        const b = results[results.len - 1];
        w("\nv2 relative to v1: put {d:.2}x, search {d:.2}x, open {d:.2}x, disk {d:.2}x\n", .{
            ratio(rate(b.vec_n, b.put_ns), rate(a.vec_n, a.put_ns)),
            ratio(@floatFromInt(a.search_ns), @floatFromInt(b.search_ns)),
            ratio(@floatFromInt(a.open_ns), @floatFromInt(b.open_ns)),
            ratio(@floatFromInt(b.bytes), @floatFromInt(a.bytes)),
        });
        w("(>1 means v2 is faster, except disk where it means v2 is bigger)\n", .{});
    }
}

/// Spelled out in the header because the persist column is meaningless without it: v1's
/// `save` cost scales with capacity rather than with what changed, so how often it is called
/// is most of the ingest number.
fn persistLabel(opts: Options) []const u8 {
    if (opts.no_persist) return "skipped";
    if (opts.persist_every == 1) return "after every document";
    return "amortized (see --persist-every)";
}

fn ratio(num: f64, den: f64) f64 {
    return if (den == 0) 0 else num / den;
}
fn rate(n: usize, ns: u64) f64 {
    if (ns == 0) return 0;
    return @as(f64, @floatFromInt(n)) * 1e9 / @as(f64, @floatFromInt(ns));
}
fn msOf(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn rssBytes() u64 {
    const ru = std.posix.getrusage(std.posix.rusage.SELF);
    const maxrss: u64 = @intCast(@max(ru.maxrss, 0));
    // Darwin reports maxrss in bytes; everyone else in kilobytes.
    return switch (builtin.os.tag) {
        .macos, .ios, .tvos, .watchos, .visionos => maxrss,
        else => maxrss * 1024,
    };
}

const ByteSize = struct {
    bytes: u64,
    pub fn format(self: ByteSize, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
        var v: f64 = @floatFromInt(self.bytes);
        var u: usize = 0;
        while (v >= 1024 and u + 1 < units.len) : (u += 1) v /= 1024;
        if (u == 0) return w.print("{d} {s}", .{ self.bytes, units[0] });
        return w.print("{d:.1} {s}", .{ v, units[u] });
    }
};
fn fmtBytes(bytes: u64) ByteSize {
    return .{ .bytes = bytes };
}

// ***************************************************************************** Argument parsing
fn parseArgs(args: []const [:0]const u8) Options {
    var o = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (eq(a, "-h") or eq(a, "--help")) usage(0);
        const val = struct {
            fn next(as: []const [:0]const u8, idx: *usize, name: []const u8) []const u8 {
                idx.* += 1;
                if (idx.* >= as.len) fatal("{s} needs a value", .{name});
                return as[idx.*];
            }
        }.next;
        if (eq(a, "--n")) {
            o.n = parseUint(val(args, &i, a));
        } else if (eq(a, "--doc")) {
            o.doc = parseUint(val(args, &i, a));
        } else if (eq(a, "--queries")) {
            o.queries = parseUint(val(args, &i, a));
        } else if (eq(a, "--persist-every")) {
            o.persist_every = parseUint(val(args, &i, a));
        } else if (eq(a, "--k")) {
            o.k = parseUint(val(args, &i, a));
        } else if (eq(a, "--seed")) {
            o.seed = parseUint(val(args, &i, a));
        } else if (eq(a, "--threshold")) {
            o.threshold = std.fmt.parseFloat(f32, val(args, &i, a)) catch
                fatal("bad --threshold", .{});
        } else if (eq(a, "--phase")) {
            const s = val(args, &i, a);
            o.phase = std.meta.stringToEnum(Phase, s) orelse
                fatal("unknown phase '{s}' (expected one of: {s})", .{ s, PHASES });
        } else if (eq(a, "--store")) {
            const s = val(args, &i, a);
            o.which = std.meta.stringToEnum(Which, s) orelse
                fatal("unknown store '{s}' (expected v1, v2, v2_indexed, both or all)", .{s});
        } else if (eq(a, "--dir")) {
            o.dir = val(args, &i, a);
        } else if (eq(a, "--no-persist")) {
            o.no_persist = true;
        } else if (eq(a, "--keep")) {
            o.keep = true;
        } else if (eq(a, "--reuse")) {
            o.reuse = true;
            // Reusing a database and then deleting it is never what anyone means.
            o.keep = true;
        } else {
            fatal("unknown argument '{s}' (try --help)", .{a});
        }
    }
    if (o.n == 0) fatal("--n must be positive", .{});
    if (o.doc == 0) fatal("--doc must be positive", .{});
    if (o.persist_every == 0) fatal("--persist-every must be positive", .{});
    return o;
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn parseUint(s: []const u8) usize {
    return std.fmt.parseUnsigned(usize, s, 10) catch fatal("'{s}' is not a number", .{s});
}
fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("storebench: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
fn usage(code: u8) noreturn {
    std.debug.print(
        \\usage: storebench [options]
        \\
        \\  --n <count>        vectors to ingest (default 100000)
        \\  --doc <count>      sentences per document, one persist call each (default 20)
        \\  --queries <count>  searches to time (default 200)
        \\  --k <count>        candidates for the indexed run (default 500)
        \\  --persist-every <n>  save/flush once every n documents (default 1)
        \\  --threshold <f>    similarity floor for search (default 0.5)
        \\  --phase <p>        ingest, search, open, or all (default all)
        \\  --store <s>        v1, v2, v2_indexed, both, all (default both)
        \\  --no-persist       skip save/flush, isolating the put path
        \\  --dir <path>       working directory (default bench-data)
        \\  --keep             leave the databases behind
        \\  --reuse            measure the databases already in --dir, skipping ingest
        \\                     (implies --keep; this is how to measure a cold open)
        \\  --seed <n>         corpus seed (default 42)
        \\
    , .{});
    std.process.exit(code);
}

const std = @import("std");
const builtin = @import("builtin");
const dve = @import("dve");
const Timer = dve.util.Timer;
const nanos = dve.util.nanos;
const Nanos = dve.util.Nanos;

/// mpnet's width, which is what the design target is sized against. The v1 store holds these
/// as `@Vector(768, f32)`, padded to 4096 bytes -- the same ~24% the v2 store pays on disk,
/// paid in RAM instead. That shows up in the rss column.
const VEC_SZ = 768;
const V1 = dve.vec_storage.Storage(VEC_SZ, f32);
const V2 = dve.vstore.VStore(VEC_SZ, f32);
const Codes = dve.codes.Codes(VEC_SZ, f32, dve.codes.DEFAULT_BITS);
const storedDotAt = dve.vec_util.storedDotAt;
