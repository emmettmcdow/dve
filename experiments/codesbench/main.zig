//! Does `src/codes.zig` do what `experiments/results/binrecall.md` promised?
//!
//! `binrecall` measured the *idea* with a throwaway encoder. This measures the module that
//! shipped, on the same corpus, so an encoder that is subtly wrong cannot hide behind
//! self-consistent unit tests. It also times the real `search` against the synthetic ceiling
//! in `hamscan.md`.
//!
//! Expected, at 384 bits on 37,689 real mpnet vectors: recall@500 ~= 0.992 on queries with no
//! near-duplicate in the corpus. Anything much below that means the module and the experiment
//! disagree, and the module is the one that is wrong.

const DIM = 768;
const TOP = 10;
/// Above this two chunks are the same text. The corpus is full of duplicated boilerplate
/// (src/chunking.md), and a query with 494 exact copies of itself measures a tie-break rather
/// than a code -- see binrecall.md. Scored separately for the same reason.
const DUP_COS: f32 = 0.99;
const KS = [_]usize{ 10, 50, 100, 200, 500, 1000 };

const Options = struct {
    dir: []const u8 = "wikitest/wikidata/wikitest-db",
    file: []const u8 = "mpnet_embedding.db",
    queries: usize = 300,
    bits: usize = 384,
    threads: ?usize = null,
    seed: u64 = 1,
};

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);
    const opts = parseArgs(args);

    var dir = std.fs.cwd().openDir(opts.dir, .{}) catch
        fatal("cannot open '{s}' (run from the repo root)", .{opts.dir});
    defer dir.close();

    var store = try Store.init(gpa, dir, .{});
    defer store.deinit();
    store.load(opts.file) catch |e| fatal("cannot load: {t}", .{e});

    var n: usize = 0;
    for (store.index) |e| {
        if (e.occupied) n += 1;
    }
    if (n == 0) fatal("database holds no vectors", .{});

    // Flat and unpadded: `store.vectors` is a []@Vector(768, f32), padded to 4096 bytes an
    // element, and the exact scan below wants the bare 3072.
    const vecs = try gpa.alloc(f32, n * DIM);
    defer gpa.free(vecs);
    var slot_of = try gpa.alloc(u32, n);
    defer gpa.free(slot_of);
    {
        var i: usize = 0;
        for (store.index, 0..) |e, s| {
            if (!e.occupied) continue;
            const v: [DIM]f32 = store.vectors[s];
            @memcpy(vecs[i * DIM ..][0..DIM], &v);
            slot_of[i] = @intCast(i); // dense: slot == index into `vecs`
            i += 1;
        }
    }

    // ---------------------------------------------------------------------------- build
    var c = try C.init(gpa, .{ .threads = opts.threads, .capacity = n });
    defer c.deinit();

    var build_timer = try std.time.Timer.start();
    for (0..n) |i| try c.put(i, @ptrCast(vecs[i * DIM ..][0..DIM]));
    const build_ns = build_timer.read();

    std.debug.print(
        \\codesbench -- src/codes.zig against the corpus binrecall measured
        \\
        \\corpus     {s}/{s}
        \\vectors    {d} live, {d} dims
        \\code       {d} bits ({d} bytes)  [module default is {d}]
        \\resident   {f} for {d} vectors -> {f} projected to 35M
        \\build      {D} ({D} per vector)
        \\queries    {d}, threads {?d}
        \\
        \\
    , .{
        opts.dir,    opts.file,
        n,           DIM,
        C.CODE_BITS, C.CODE_BYTES,
        dve.codes.DEFAULT_BITS,
        fmtBytes(c.bytes()), n,
        fmtBytes(C.CODE_BYTES * 35_000_000),
        build_ns,    build_ns / n,
        opts.queries, opts.threads,
    });

    // ------------------------------------------------------------------------ ground truth
    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rng = prng.random();
    const queries = try gpa.alloc(u32, opts.queries);
    defer gpa.free(queries);
    for (queries) |*q| q.* = rng.uintLessThan(u32, @intCast(n));

    const truth = try gpa.alloc([TOP]u32, opts.queries);
    defer gpa.free(truth);
    const dup = try gpa.alloc(bool, opts.queries);
    defer gpa.free(dup);
    var dup_n: usize = 0;
    for (queries, truth, dup) |q, *t, *d| {
        d.* = exactTop(vecs, n, q, t) > DUP_COS;
        if (d.*) dup_n += 1;
    }
    std.debug.print("{d} of {d} queries have a near-duplicate; scored separately\n\n", .{
        dup_n, opts.queries,
    });

    // ------------------------------------------------------------------------------ recall
    const buf = try gpa.alloc(C.Candidate, KS[KS.len - 1]);
    defer gpa.free(buf);

    std.debug.print("{s:>8}{s:>12}{s:>12}{s:>14}\n", .{ "K", "recall", "no-dup", "scan ms" });
    std.debug.print("{s:->8}{s:->12}{s:->12}{s:->14}\n", .{ "", "", "", "" });

    for (KS) |k| {
        var hit: f64 = 0;
        var clean: f64 = 0;
        var clean_n: usize = 0;
        var best_ns: u64 = std.math.maxInt(u64);

        for (queries, truth, dup) |q, t, is_dup| {
            var timer = try std.time.Timer.start();
            const found = try c.search(@ptrCast(vecs[@as(usize, q) * DIM ..][0..DIM]), buf[0..k]);
            best_ns = @min(best_ns, timer.read());

            // The query is its own nearest neighbour at distance 0; ground truth excludes it,
            // so the candidate list has to as well or K is effectively K-1.
            var h: f64 = 0;
            for (t) |id| {
                for (buf[0..found]) |cand| {
                    if (cand.slot == id) {
                        h += 1;
                        break;
                    }
                }
            }
            hit += h / TOP;
            if (!is_dup) {
                clean += h / TOP;
                clean_n += 1;
            }
        }
        const nq: f64 = @floatFromInt(opts.queries);
        std.debug.print("{d:>8}{d:>12.3}{d:>12.3}{d:>14.3}\n", .{
            k,
            hit / nq,
            clean / @as(f64, @floatFromInt(@max(clean_n, 1))),
            @as(f64, @floatFromInt(best_ns)) / 1e6,
        });
    }

    std.debug.print(
        \\
        \\"no-dup" is the honest column; see experiments/results/binrecall.md for why. It should
        \\land near 0.992 at K=500 for a {d}-bit code. "scan ms" is the fastest of the {d}
        \\queries, and at this corpus size the array is cache-resident -- hamscan.md has the
        \\rate that matters at 35M.
        \\
    , .{ C.CODE_BITS, opts.queries });
}

/// Exact cosine top-`TOP`, excluding the query. Stored vectors are L2 normalized, so the dot
/// product is the cosine. Returns the best similarity, which is how a duplicate is spotted.
fn exactTop(vecs: []const f32, n: usize, q: u32, out: *[TOP]u32) f32 {
    var best_sim = [_]f32{-2.0} ** TOP;
    var best_id = [_]u32{0} ** TOP;
    const qv = vecs[@as(usize, q) * DIM ..][0..DIM];
    for (0..n) |i| {
        if (i == q) continue;
        const sim = dot(qv, vecs[i * DIM ..][0..DIM]);
        if (sim <= best_sim[TOP - 1]) continue;
        var j: usize = TOP - 1;
        while (j > 0 and best_sim[j - 1] < sim) : (j -= 1) {
            best_sim[j] = best_sim[j - 1];
            best_id[j] = best_id[j - 1];
        }
        best_sim[j] = sim;
        best_id[j] = @intCast(i);
    }
    out.* = best_id;
    return best_sim[0];
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
    return @reduce(.Add, acc);
}

const ByteSize = struct {
    bytes: u64,
    pub fn format(self: ByteSize, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const units = [_][]const u8{ "B", "KB", "MB", "GB" };
        var v: f64 = @floatFromInt(self.bytes);
        var u: usize = 0;
        while (v >= 1024 and u + 1 < units.len) : (u += 1) v /= 1024;
        if (u == 0) return w.print("{d} {s}", .{ self.bytes, units[0] });
        return w.print("{d:.1} {s}", .{ v, units[u] });
    }
};
fn fmtBytes(b: u64) ByteSize {
    return .{ .bytes = b };
}

fn parseArgs(args: [][:0]u8) Options {
    var o = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (eq(a, "-h") or eq(a, "--help")) usage(0);
        i += 1;
        if (i >= args.len) fatal("{s} needs a value", .{a});
        const v = args[i];
        if (eq(a, "--dir")) o.dir = v
        else if (eq(a, "--file")) o.file = v
        else if (eq(a, "--queries")) o.queries = parseUint(v)
        else if (eq(a, "--threads")) o.threads = parseUint(v)
        else if (eq(a, "--seed")) o.seed = parseUint(v)
        else fatal("unknown argument '{s}' (try --help)", .{a});
    }
    return o;
}
fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn parseUint(s: []const u8) usize {
    return std.fmt.parseUnsigned(usize, s, 10) catch fatal("'{s}' is not a number", .{s});
}
fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("codesbench: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
fn usage(code: u8) noreturn {
    std.debug.print(
        \\usage: codesbench [options]
        \\
        \\  --dir <path>       database directory (default wikitest/wikidata/wikitest-db)
        \\  --file <name>      database file (default mpnet_embedding.db)
        \\  --queries <count>  queries to average over (default 300)
        \\  --threads <count>  scan threads (default: the pool's own choice)
        \\  --seed <n>         rng seed (default 1)
        \\
    , .{});
    std.process.exit(code);
}

const std = @import("std");
const dve = @import("dve");
const Store = dve.vec_storage.Storage(DIM, f32);
const C = dve.codes.Codes(DIM, f32, 384);
