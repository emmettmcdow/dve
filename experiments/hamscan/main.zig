//! How fast can we scan 1-bit codes in RAM?
//!
//! `binrecall` measured this incidentally at 27.8 GB/s and the number was worthless: its
//! 37,689 codes are 3.6 MB, and this machine's L2 is 6 MB, so it measured cache. The planned
//! codes cache for Simple Wikipedia is ~3.4 GB. This sweeps from well inside cache to well
//! past it so the DRAM-bound rate -- the one that decides whether search is "near instant" --
//! is measured rather than extrapolated.
//!
//! Two inner loops, because it is not obvious which resource binds. A 96-byte code is 12
//! u64 XOR+popcounts, and on AArch64 `@popCount` on a u64 has to move to a vector register,
//! `cnt`, and `addv` back -- three or four instructions for eight bytes. The NEON form does
//! 16 bytes per `cnt` with no round trip. If the two rates differ, the scan is compute-bound
//! and the code layout matters; if they agree, it is bandwidth-bound and only the array size
//! does.
//!
//! The work is a real top-K selection, not a checksum. A bare distance loop would be
//! optimised into something the real search cannot use, and the rejection branch is part of
//! what the scan costs.

const DIM = 768;
const CODE_BYTES = DIM / 8; // 96
const CODE_WORDS = CODE_BYTES / 8; // 12 u64
const CODE_LANES = CODE_BYTES / 16; // 6 NEON registers

const Impl = enum {
    u64x12,
    neon,
    /// Control, not a candidate. Identical loads and identical access pattern, but no
    /// popcount and no per-code reduction -- just an accumulate. It measures what this
    /// machine will stream at, which is the only way to tell whether the two real loops are
    /// at the memory wall or merely at their own compute ceiling.
    stream,
};

const Options = struct {
    /// Largest corpus to scan. The array is allocated once at this size and shorter runs
    /// scan a prefix, so the sweep costs one allocation rather than one per size.
    n: usize = 35_000_000,
    queries: usize = 5,
    k: usize = 100,
    threads: usize = 0, // 0 = sweep
    seed: u64 = 3,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const opts = parseArgs(args);

    const cores = std.Thread.getCpuCount() catch 8;

    std.debug.print(
        \\hamming scan throughput -- {d}-bit codes, {d} bytes each
        \\
        \\corpus      up to {d} codes ({f})
        \\queries     {d} per configuration, top-{d} kept
        \\machine     {d} logical cores
        \\
        \\allocating and filling...
    , .{ DIM, CODE_BYTES, opts.n, fmtBytes(opts.n * CODE_BYTES), opts.queries, opts.k, cores });

    const codes = try gpa.alignedAlloc(u64, .@"64", opts.n * CODE_WORDS);
    defer gpa.free(codes);
    {
        // Filled with uniform random bits. The content does not change popcount's cost, and
        // uniform codes make the top-K threshold converge the way real ones do -- distances
        // cluster near DIM/2, so after a few hundred candidates almost everything is rejected
        // by a single comparison, which is what the real scan does too.
        var prng = std.Random.DefaultPrng.init(opts.seed);
        prng.random().bytes(std.mem.sliceAsBytes(codes));
    }
    std.debug.print(" done\n\n", .{});

    var thread_counts: [8]usize = undefined;
    var n_tc: usize = 0;
    if (opts.threads != 0) {
        thread_counts[0] = opts.threads;
        n_tc = 1;
    } else {
        var t: usize = 1;
        while (t <= cores) : (t *= 2) {
            thread_counts[n_tc] = t;
            n_tc += 1;
        }
        if (thread_counts[n_tc - 1] != cores) {
            thread_counts[n_tc] = cores;
            n_tc += 1;
        }
    }

    const sizes = [_]usize{ 100_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, 35_000_000 };

    for (std.enums.values(Impl)) |impl| {
        std.debug.print("=== {s} ===\n", .{@tagName(impl)});
        std.debug.print("{s:>12}{s:>11}", .{ "codes", "bytes" });
        for (thread_counts[0..n_tc]) |t| {
            var buf: [16]u8 = undefined;
            std.debug.print("{s:>14}", .{std.fmt.bufPrint(&buf, "{d}t ms", .{t}) catch "?"});
        }
        std.debug.print("{s:>12}\n", .{"best GB/s"});
        std.debug.print("{s:->12}{s:->11}", .{ "", "" });
        for (thread_counts[0..n_tc]) |_| std.debug.print("{s:->14}", .{""});
        std.debug.print("{s:->12}\n", .{""});

        for (sizes) |n| {
            if (n > opts.n) continue;
            const bytes = n * CODE_BYTES;
            var bbuf: [24]u8 = undefined;
            const btxt = std.fmt.bufPrint(&bbuf, "{f}", .{fmtBytes(bytes)}) catch "?";
            std.debug.print("{d:>12}{s:>11}", .{ n, btxt });
            var best_rate: f64 = 0;
            for (thread_counts[0..n_tc]) |t| {
                const ns = try timeScan(gpa, io, codes, n, opts, impl, t);
                const ms = @as(f64, @floatFromInt(ns)) / 1e6;
                const rate = @as(f64, @floatFromInt(bytes)) / (@as(f64, @floatFromInt(ns)) / 1e9) / 1e9;
                best_rate = @max(best_rate, rate);
                std.debug.print("{d:>14.2}", .{ms});
            }
            std.debug.print("{d:>12.1}\n", .{best_rate});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print(
        \\Each cell is the fastest of {d} queries against the whole array: scan every code, keep
        \\the best {d}. The minimum rather than the mean -- scheduler noise on a 4+6 heterogeneous
        \\CPU only ever makes a run slower, so the minimum is the machine and the mean is the
        \\machine plus whatever else was running. "best GB/s" is over the thread counts in a row.
        \\
        \\L2 on this machine is 6 MB, so the 100k row (9.6 MB) is near-resident and the rows
        \\below it are not. Read the large rows; the small one is there to show the cliff.
        \\
    , .{ opts.queries, opts.k });
}

// ************************************************************************************ Scanning
/// Bounded top-K. Insertion into a sorted array rather than a heap: the `d >= worst` reject
/// above it means a push happens about K*ln(N/K) times in a scan of N, which is ~1,300 for
/// 35M codes, so the insert cost is irrelevant and the branch predictor sees one outcome.
const TopK = struct {
    dist: []u16,
    id: []u32,
    n: usize = 0,
    worst: u32 = std.math.maxInt(u32),

    fn init(buf_d: []u16, buf_i: []u32) TopK {
        return .{ .dist = buf_d, .id = buf_i };
    }

    fn push(self: *TopK, id: u32, d: u32) void {
        const k = self.dist.len;
        var j = @min(self.n, k - 1);
        while (j > 0 and self.dist[j - 1] > d) : (j -= 1) {
            self.dist[j] = self.dist[j - 1];
            self.id[j] = self.id[j - 1];
        }
        self.dist[j] = @intCast(d);
        self.id[j] = id;
        if (self.n < k) self.n += 1;
        if (self.n == k) self.worst = self.dist[k - 1];
    }
};

fn scanU64(codes: []const u64, base: u32, q: *const [CODE_WORDS]u64, top: *TopK) void {
    var i: usize = 0;
    var slot: u32 = base;
    while (i < codes.len) : (i += CODE_WORDS) {
        const c = codes[i..][0..CODE_WORDS];
        var d: u32 = 0;
        inline for (0..CODE_WORDS) |w| d += @popCount(q[w] ^ c[w]);
        if (d < top.worst) top.push(slot, d);
        slot += 1;
    }
}

const Lane = @Vector(16, u8);

fn scanStream(codes: []const u64, base: u32, q: *const [CODE_WORDS]u64, top: *TopK) void {
    const qb: *const [CODE_BYTES]u8 = @ptrCast(q);
    var ql: [CODE_LANES]Lane = undefined;
    inline for (0..CODE_LANES) |l| ql[l] = qb[l * 16 ..][0..16].*;

    const bytes = std.mem.sliceAsBytes(codes);
    var acc: Lane = @splat(0);
    var off: usize = 0;
    while (off < bytes.len) : (off += CODE_BYTES) {
        inline for (0..CODE_LANES) |l| {
            const cv: Lane = bytes[off + l * 16 ..][0..16].*;
            acc +%= cv ^ ql[l];
        }
    }
    // One observable result for the whole chunk, so nothing is optimised away, but no
    // per-code dependency chain.
    const wide: @Vector(16, u16) = acc;
    top.push(base, @reduce(.Add, wide) % 769);
}

fn scanNeon(codes: []const u64, base: u32, q: *const [CODE_WORDS]u64, top: *TopK) void {
    const qb: *const [CODE_BYTES]u8 = @ptrCast(q);
    var ql: [CODE_LANES]Lane = undefined;
    inline for (0..CODE_LANES) |l| ql[l] = qb[l * 16 ..][0..16].*;

    const bytes = std.mem.sliceAsBytes(codes);
    var off: usize = 0;
    var slot: u32 = base;
    while (off < bytes.len) : (off += CODE_BYTES) {
        // Six 16-byte lanes, each XOR'd then popcounted in one `cnt`. The per-lane counts max
        // out at 8, so six of them sum to at most 48 and the u8 accumulator cannot overflow --
        // but the 16-lane *reduction* would (16 x 48 = 768), so it widens first.
        var acc: Lane = @splat(0);
        inline for (0..CODE_LANES) |l| {
            const cv: Lane = bytes[off + l * 16 ..][0..16].*;
            acc +%= @popCount(cv ^ ql[l]);
        }
        const wide: @Vector(16, u16) = acc;
        const d: u32 = @reduce(.Add, wide);
        if (d < top.worst) top.push(slot, d);
        slot += 1;
    }
}

/// Threads pull fixed-size chunks off an atomic counter rather than taking an equal slice
/// each. This machine is 4 performance cores and 6 efficiency cores, and an equal split puts
/// the same work on both kinds: the join then waits for whichever slice landed on an E-core,
/// which showed up as a 4-thread column *slower* than the 8-thread one and a single-threaded
/// 35M figure six times its own trend. Chunks small enough to rebalance and large enough that
/// the atomic disappears fix it, and are what a real implementation would do anyway.
const CHUNK_CODES: usize = 32768; // 3 MB of codes, ~1,070 chunks at 35M

const Shared = struct {
    codes: []const u64,
    n: usize,
    q: *const [CODE_WORDS]u64,
    impl: Impl,
    k: usize,
    next: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
};

const Job = struct {
    shared: *Shared,
    /// Kept so the merge below has something to do, which stops the optimiser deciding the
    /// scan had no observable effect.
    out_best: u32 = 0,

    fn run(self: *Job) void {
        const sh = self.shared;
        var buf_d: [8192]u16 = undefined;
        var buf_i: [8192]u32 = undefined;
        var top = TopK.init(buf_d[0..sh.k], buf_i[0..sh.k]);

        while (true) {
            const c = sh.next.fetchAdd(1, .monotonic);
            const start = c * CHUNK_CODES;
            if (start >= sh.n) break;
            const end = @min(start + CHUNK_CODES, sh.n);
            const slice = sh.codes[start * CODE_WORDS .. end * CODE_WORDS];
            switch (sh.impl) {
                .u64x12 => scanU64(slice, @intCast(start), sh.q, &top),
                .neon => scanNeon(slice, @intCast(start), sh.q, &top),
                .stream => scanStream(slice, @intCast(start), sh.q, &top),
            }
        }
        self.out_best = if (top.n > 0) top.dist[0] else 0;
    }
};

fn timeScan(
    gpa: std.mem.Allocator,
    io: std.Io,
    codes: []const u64,
    n: usize,
    opts: Options,
    impl: Impl,
    threads: usize,
) !u64 {
    const jobs = try gpa.alloc(Job, threads);
    defer gpa.free(jobs);
    const handles = try gpa.alloc(std.Thread, threads);
    defer gpa.free(handles);

    var prng = std.Random.DefaultPrng.init(opts.seed +% 99);
    const rng = prng.random();

    var best: u64 = std.math.maxInt(u64);
    var sink: u64 = 0;
    for (0..opts.queries) |_| {
        // A fresh random query each time, so no run benefits from the previous one's
        // converged threshold.
        var q: [CODE_WORDS]u64 = undefined;
        rng.bytes(std.mem.asBytes(&q));

        var shared = Shared{
            .codes = codes,
            .n = n,
            .q = &q,
            .impl = impl,
            .k = opts.k,
        };
        const started: std.Io.Timestamp = .now(io, .awake);
        for (jobs) |*job| job.* = .{ .shared = &shared };
        for (handles, jobs) |*h, *job| h.* = try std.Thread.spawn(.{}, Job.run, .{job});
        for (handles) |h| h.join();
        best = @min(best, @as(u64, @intCast(started.untilNow(io, .awake).toNanoseconds())));
        for (jobs) |job| sink +%= job.out_best;
    }
    std.mem.doNotOptimizeAway(sink);
    return best;
}

// ******************************************************************************* Presentation
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

fn parseArgs(args: []const [:0]const u8) Options {
    var o = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (eq(a, "-h") or eq(a, "--help")) usage(0);
        i += 1;
        if (i >= args.len) fatal("{s} needs a value", .{a});
        const v = args[i];
        if (eq(a, "--n")) o.n = parseUint(v)
        else if (eq(a, "--queries")) o.queries = parseUint(v)
        else if (eq(a, "--k")) o.k = parseUint(v)
        else if (eq(a, "--threads")) o.threads = parseUint(v)
        else if (eq(a, "--seed")) o.seed = parseUint(v)
        else fatal("unknown argument '{s}' (try --help)", .{a});
    }
    if (o.k == 0 or o.k > 8192) fatal("--k must be 1..8192", .{});
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
    std.debug.print("hamscan: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
fn usage(code: u8) noreturn {
    std.debug.print(
        \\usage: hamscan [options]
        \\
        \\  --n <count>        largest corpus to scan (default 35000000, ~3.1 GiB)
        \\  --queries <count>  queries averaged per configuration (default 5)
        \\  --k <count>        candidates kept per query, 1..1024 (default 100)
        \\  --threads <count>  fix the thread count instead of sweeping
        \\  --seed <n>         rng seed (default 3)
        \\
    , .{});
    std.process.exit(code);
}

const std = @import("std");
