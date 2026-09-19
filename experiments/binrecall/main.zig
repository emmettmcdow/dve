//! Binary recall: does a 1-bit code surface the vectors that real cosine says are the answer?
//!
//! The planned search is two stages -- scan 1-bit codes in RAM for ~K candidates, then read
//! those K full vectors from disk and rank them by exact cosine. Stage 2 can only *reorder*
//! what stage 1 hands it. Anything stage 1 misses is gone. So the whole design rests on one
//! number: what fraction of the true top-10 by cosine lands inside the Hamming top-K?
//!
//! This reads a corpus of real mpnet embeddings out of an existing `vec_storage.zig` database
//! and measures that directly. It builds no index and writes nothing.
//!
//! **"1-bit quantization" is not one thing**, which is why there are four variants here. The
//! LSH that actually approximates cosine is SimHash: for a random unit vector r,
//! `P(sign(r.x) != sign(r.y)) = theta/pi` exactly. Taking the sign of each *coordinate* is
//! SimHash with the axis-aligned basis substituted for a random one, which is only sound if
//! the embedding's coordinates are isotropic. Transformer embeddings are not -- they occupy a
//! narrow cone with a strong mean direction, so a coordinate whose sign is the same for 95% of
//! the corpus spends a bit of the code on nothing.
//!
//! Note `centered` and `centered_simhash` measure angular distance in the *centered* space,
//! which is not the metric stage 2 ranks by. They can be better search and still score worse
//! here, because the yardstick is exact cosine on the raw vectors.

const Variant = enum {
    /// bit i = (x[i] > 0). The naive reading of "1-bit quantization".
    sign,
    /// bit i = (x[i] > mean[i]), the corpus component-wise mean. Costs 3 KB of parameters that
    /// drift as the corpus grows.
    centered,
    /// bit i = (R.x)[i] > 0 for a fixed random gaussian R. No learned parameters; costs a
    /// 768x768 matvec per vector at ingest.
    simhash,
    /// SimHash over the centered vectors, which is the textbook-correct form when the
    /// distribution has a mean offset.
    centered_simhash,
};

const Options = struct {
    dir: []const u8 = "wikitest/wikidata/wikitest-db",
    file: []const u8 = "mpnet_embedding.db",
    queries: usize = 300,
    /// Subsample the corpus. Recall degrades as vectors crowd into the same Hamming ball, so
    /// a result at one size says little about another; two sizes at least show the direction.
    limit: usize = 0,
    seed: u64 = 1,
    verbose: bool = false,
};

const DIM = 768;
const CODE_BYTES = DIM / 8; // 96
const CODE_WORDS = CODE_BYTES / 8; // 12 u64s
const TOP = 10;
const KS = [_]usize{ 10, 25, 50, 100, 200, 500, 1000, 2000 };

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);
    const opts = parseArgs(args);

    // ------------------------------------------------------------------- load the corpus
    var dir = std.fs.cwd().openDir(opts.dir, .{}) catch
        fatal("cannot open '{s}' (run from the repo root)", .{opts.dir});
    defer dir.close();

    var store = try Store.init(gpa, dir, .{});
    defer store.deinit();
    store.load(opts.file) catch |e|
        fatal("cannot load '{s}/{s}': {t}", .{ opts.dir, opts.file, e });

    // Copied out of the store into a flat, unpadded array. `vectors` is a []@Vector(768, f32),
    // which zig pads to 4096 bytes an element; the exact scan below wants 3072.
    var live: usize = 0;
    for (store.index) |e| {
        if (e.occupied) live += 1;
    }
    if (live == 0) fatal("database holds no vectors", .{});

    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rng = prng.random();

    const n = if (opts.limit == 0) live else @min(opts.limit, live);
    const vecs = try gpa.alloc(f32, n * DIM);
    defer gpa.free(vecs);
    {
        // Reservoir-free subsample: take every `stride`-th live vector, which keeps the
        // document mix rather than the first N documents' vectors.
        const stride = live / n;
        var taken: usize = 0;
        var seen: usize = 0;
        for (store.index, 0..) |e, i| {
            if (!e.occupied) continue;
            defer seen += 1;
            if (seen % stride != 0 or taken >= n) continue;
            const v: [DIM]f32 = store.vectors[i];
            @memcpy(vecs[taken * DIM ..][0..DIM], &v);
            taken += 1;
        }
        if (taken != n) fatal("subsample took {d} of {d}", .{ taken, n });
    }

    std.debug.print(
        \\binary recall -- does a 1-bit code surface what cosine says is the answer?
        \\
        \\corpus     {s}/{s}
        \\vectors    {d} live, {d} sampled, {d} dims
        \\queries    {d}, drawn from the corpus, each excluded from its own results
        \\ground truth: exact cosine top-{d}
        \\
        \\
    , .{ opts.dir, opts.file, live, n, DIM, opts.queries, TOP });

    // ------------------------------------------------------------------- shared parameters
    const mean = try gpa.alloc(f32, DIM);
    defer gpa.free(mean);
    componentMean(vecs, n, mean);

    // One rotation, shared by both simhash variants so they differ only in the centering.
    const rot = try gpa.alloc(f32, DIM * DIM);
    defer gpa.free(rot);
    for (rot) |*r| r.* = rng.floatNorm(f32);

    const queries = try gpa.alloc(u32, opts.queries);
    defer gpa.free(queries);
    for (queries) |*q| q.* = rng.uintLessThan(u32, @intCast(n));

    // Ground truth is the same for every variant, so it is computed once.
    const truth = try gpa.alloc([TOP]u32, opts.queries);
    defer gpa.free(truth);
    var truth_timer = try std.time.Timer.start();
    for (queries, truth) |q, *t| exactTop(vecs, n, q, t);
    const truth_ns = truth_timer.read();
    std.debug.print("exact top-{d}: {D} for {d} queries ({D} each)\n\n", .{
        TOP, truth_ns, opts.queries, truth_ns / opts.queries,
    });

    // ------------------------------------------------------------------------- the variants
    printHeader();
    for (std.enums.values(Variant)) |variant| {
        const codes = try gpa.alignedAlloc(u64, .@"8", n * CODE_WORDS);
        defer gpa.free(codes);
        try encode(gpa, variant, vecs, n, mean, rot, codes);
        const stats = try recallOf(gpa, codes, n, queries, truth);
        printRow(variant, stats, opts.verbose);
    }

    std.debug.print(
        \\
        \\recall@K is the share of the exact top-{d} found inside the Hamming top-K. Ties at the
        \\K-th distance are counted fractionally, so these are expected values rather than one
        \\arbitrary tie-break.
        \\
        \\"dead bits" are code positions whose sign is the same for over 95% of the corpus --
        \\bits spent carrying nothing. "balance" is the mean over bits of |P(set) - 0.5|, where
        \\0 is a perfectly balanced code and 0.5 is a constant one.
        \\
    , .{TOP});
}

// ************************************************************************************ Encoding
fn componentMean(vecs: []const f32, n: usize, out: []f32) void {
    @memset(out, 0);
    for (0..n) |i| {
        const v = vecs[i * DIM ..][0..DIM];
        for (out, v) |*m, x| m.* += x;
    }
    const inv = 1.0 / @as(f32, @floatFromInt(n));
    for (out) |*m| m.* *= inv;
}

fn encode(
    gpa: std.mem.Allocator,
    variant: Variant,
    vecs: []const f32,
    n: usize,
    mean: []const f32,
    rot: []const f32,
    out: []u64,
) !void {
    const rotated = variant == .simhash or variant == .centered_simhash;
    const centered = variant == .centered or variant == .centered_simhash;

    const work = try gpa.alloc(f32, DIM);
    defer gpa.free(work);

    for (0..n) |i| {
        const v = vecs[i * DIM ..][0..DIM];
        // Centering happens before rotation: rotating first and subtracting a mean in the
        // rotated space would be the same thing, but this way one mean serves both variants.
        if (centered) {
            for (work, v, mean) |*w, x, m| w.* = x - m;
        } else {
            @memcpy(work, v);
        }

        const code = out[i * CODE_WORDS ..][0..CODE_WORDS];
        @memset(code, 0);
        for (0..DIM) |b| {
            const proj = if (rotated) dot(rot[b * DIM ..][0..DIM], work) else work[b];
            if (proj > 0) code[b / 64] |= @as(u64, 1) << @intCast(b % 64);
        }
    }
}

fn dot(a: []const f32, b: []const f32) f32 {
    const LANES = 16;
    var acc: @Vector(LANES, f32) = @splat(0);
    var i: usize = 0;
    while (i + LANES <= a.len) : (i += LANES) {
        const va: @Vector(LANES, f32) = a[i..][0..LANES].*;
        const vb: @Vector(LANES, f32) = b[i..][0..LANES].*;
        acc += va * vb;
    }
    var total = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) total += a[i] * b[i];
    return total;
}

// ******************************************************************************* Ground truth
/// Exact cosine top-`TOP`, excluding the query itself. Every vector in the store is L2
/// normalized -- `validate` enforces it -- so the dot product is the cosine.
fn exactTop(vecs: []const f32, n: usize, q: u32, out: *[TOP]u32) void {
    var best_sim = [_]f32{-2.0} ** TOP;
    var best_id = [_]u32{0} ** TOP;
    const qv = vecs[@as(usize, q) * DIM ..][0..DIM];

    for (0..n) |i| {
        if (i == q) continue;
        const sim = dot(qv, vecs[i * DIM ..][0..DIM]);
        if (sim <= best_sim[TOP - 1]) continue;
        // Insertion into a 10-element sorted list: cheaper than a heap at this size, and the
        // scan above is what dominates anyway.
        var j: usize = TOP - 1;
        while (j > 0 and best_sim[j - 1] < sim) : (j -= 1) {
            best_sim[j] = best_sim[j - 1];
            best_id[j] = best_id[j - 1];
        }
        best_sim[j] = sim;
        best_id[j] = @intCast(i);
    }
    out.* = best_id;
}

// ************************************************************************************* Recall
const Stats = struct {
    recall: [KS.len]f64,
    /// Mean Hamming distance from a query to its exact nearest neighbour, in bits. A code that
    /// has collapsed puts everything at the same distance and this goes to ~DIM/2.
    mean_nn_dist: f64,
    dead_bits: usize,
    balance: f64,
    scan_ns: u64,
};

fn recallOf(
    gpa: std.mem.Allocator,
    codes: []const u64,
    n: usize,
    queries: []const u32,
    truth: []const [TOP]u32,
) !Stats {
    const dists = try gpa.alloc(u16, n);
    defer gpa.free(dists);
    var hist = try gpa.alloc(u32, DIM + 1);
    defer gpa.free(hist);

    var sum: [KS.len]f64 = @splat(0);
    var nn_sum: f64 = 0;
    var scan_ns: u64 = 0;
    var timer = try std.time.Timer.start();

    for (queries, truth) |q, t| {
        const qc = codes[@as(usize, q) * CODE_WORDS ..][0..CODE_WORDS];
        @memset(hist, 0);

        timer.reset();
        for (0..n) |i| {
            const c = codes[i * CODE_WORDS ..][0..CODE_WORDS];
            var d: u32 = 0;
            inline for (0..CODE_WORDS) |w| d += @popCount(qc[w] ^ c[w]);
            dists[i] = @intCast(d);
        }
        scan_ns += timer.read();

        for (0..n) |i| {
            if (i == q) continue; // the query is not its own result
            hist[dists[i]] += 1;
        }
        nn_sum += @floatFromInt(dists[t[0]]);

        for (KS, 0..) |k, ki| {
            // Everything strictly closer than `dstar` is inside the top-K for certain;
            // `frac` of the items *at* dstar are, which is how ties are shared out.
            var cum: usize = 0;
            var dstar: usize = 0;
            while (dstar <= DIM and cum + hist[dstar] < k) : (dstar += 1) cum += hist[dstar];
            const at = if (dstar <= DIM) hist[dstar] else 0;
            const frac: f64 = if (at == 0) 0 else
                @as(f64, @floatFromInt(k - cum)) / @as(f64, @floatFromInt(at));

            var hit: f64 = 0;
            for (t) |id| {
                const d = dists[id];
                if (d < dstar) hit += 1 else if (d == dstar) hit += frac;
            }
            sum[ki] += hit / @as(f64, TOP);
        }
    }

    var out = Stats{
        .recall = undefined,
        .mean_nn_dist = nn_sum / @as(f64, @floatFromInt(queries.len)),
        .dead_bits = 0,
        .balance = 0,
        .scan_ns = scan_ns / queries.len,
    };
    for (&out.recall, sum) |*r, s| r.* = s / @as(f64, @floatFromInt(queries.len));

    // Bit occupancy, which is what explains a bad recall rather than merely reporting it.
    for (0..DIM) |b| {
        var set: usize = 0;
        for (0..n) |i| {
            if (codes[i * CODE_WORDS + b / 64] >> @intCast(b % 64) & 1 == 1) set += 1;
        }
        const p = @as(f64, @floatFromInt(set)) / @as(f64, @floatFromInt(n));
        if (p > 0.95 or p < 0.05) out.dead_bits += 1;
        out.balance += @abs(p - 0.5);
    }
    out.balance /= DIM;
    return out;
}

// ************************************************************************************* Report
fn printHeader() void {
    std.debug.print("{s:<17}", .{"variant"});
    for (KS) |k| std.debug.print("{d:>8}", .{k});
    std.debug.print("{s:>10}{s:>8}{s:>10}\n", .{ "dead", "bal", "scan" });
    std.debug.print("{s:<17}", .{""});
    for (KS) |_| std.debug.print("{s:>8}", .{""});
    std.debug.print("{s:>10}{s:>8}{s:>10}\n", .{ "bits", "", "ms" });
    std.debug.print("{s:-<17}", .{""});
    for (KS) |_| std.debug.print("{s:->8}", .{""});
    std.debug.print("{s:->10}{s:->8}{s:->10}\n", .{ "", "", "" });
}

fn printRow(v: Variant, s: Stats, verbose: bool) void {
    std.debug.print("{s:<17}", .{@tagName(v)});
    for (s.recall) |r| std.debug.print("{d:>8.3}", .{r});
    std.debug.print("{d:>10}{d:>8.3}{d:>10.2}\n", .{
        s.dead_bits, s.balance, @as(f64, @floatFromInt(s.scan_ns)) / 1e6,
    });
    if (verbose) {
        std.debug.print("{s:<17}mean Hamming distance to the true nearest neighbour: {d:.1} bits\n", .{ "", s.mean_nn_dist });
    }
}

// ***************************************************************************** Argument parsing
fn parseArgs(args: [][:0]u8) Options {
    var o = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (eq(a, "-h") or eq(a, "--help")) usage(0);
        if (eq(a, "--verbose")) {
            o.verbose = true;
            continue;
        }
        i += 1;
        if (i >= args.len) fatal("{s} needs a value", .{a});
        const v = args[i];
        if (eq(a, "--dir")) o.dir = v
        else if (eq(a, "--file")) o.file = v
        else if (eq(a, "--queries")) o.queries = parseUint(v)
        else if (eq(a, "--limit")) o.limit = parseUint(v)
        else if (eq(a, "--seed")) o.seed = parseUint(v)
        else fatal("unknown argument '{s}' (try --help)", .{a});
    }
    if (o.queries == 0) fatal("--queries must be positive", .{});
    return o;
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn parseUint(s: []const u8) usize {
    return std.fmt.parseUnsigned(usize, s, 10) catch fatal("'{s}' is not a number", .{s});
}
fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("binrecall: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
fn usage(code: u8) noreturn {
    std.debug.print(
        \\usage: binrecall [options]
        \\
        \\  --dir <path>       database directory (default wikitest/wikidata/wikitest-db)
        \\  --file <name>      database file (default mpnet_embedding.db)
        \\  --queries <count>  queries to average over (default 300)
        \\  --limit <count>    subsample the corpus to this many vectors (default: all)
        \\  --seed <n>         rng seed (default 1)
        \\  --verbose          also print Hamming distance to the true nearest neighbour
        \\
    , .{});
    std.process.exit(code);
}

const std = @import("std");
const dve = @import("dve");
/// The corpus lives in a `vec_storage.zig` database -- the pre-cutover format, which is why
/// that store is still in the tree. Nothing here depends on it beyond reading the bytes.
const Store = dve.vec_storage.Storage(DIM, f32);
