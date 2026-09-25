//! In-memory quantized codes: stage one of the two-stage search.
//!
//! One 1-bit code per vector, scanned linearly in RAM to narrow a corpus of millions to a few
//! hundred candidates. Stage two reads those candidates' full vectors from disk and ranks them
//! by exact cosine; this module never touches a disk and never learns that one exists.
//!
//! **It knows nothing about storage.** `vstore.zig` owns bytes, this owns codes, and neither
//! imports the other -- `vector.zig` holds both and wires them together. The only thing shared
//! is the *slot*: a code lives at `codes[slot]` for the same slot the store put the vector in.
//! That costs nothing (the store already numbers slots and already reuses them) and it means
//! there is no id map to keep consistent, no compaction pass, and no second allocation that
//! grows with the corpus rather than with its holes.
//!
//! Every constant here was measured. See `experiments/results/binrecall.md` for the code and
//! its width, `experiments/results/hamscan.md` for the scan.
//!
//! ## The code is the sign of the first `code_bits` components
//!
//! No centering, no random projection, no trained codebook. Four variants were measured
//! against exact cosine on real mpnet embeddings and at the operating point that matters --
//! 384 bits, K=500 -- plain truncated sign scored 0.992 and every variant that costs something
//! scored the same or worse. Centering wins by 1.4 points at K=100 and nothing at K=500, so
//! the 3 KB of corpus mean it needs, which drifts as documents arrive, buys nothing where it
//! would be used.
//!
//! ## Why 384 bits rather than 768
//!
//! The scan is purely memory-bound -- an inner loop with the popcount deleted entirely runs at
//! the same speed -- so scan time is exactly proportional to code width, while the candidate
//! count K is paid for separately in disk reads. That makes the two tradeable, and the trade
//! is lopsided. At 35M vectors:
//!
//!     768 bits, K=100 -> recall 0.987, 124 ms scan +  6 ms disk = 130 ms, 3.36 GB
//!     384 bits, K=500 -> recall 0.992,  62 ms scan + 32 ms disk =  94 ms, 1.68 GB
//!
//! 768 bits at K=100 is not on the Pareto frontier at all. Recall lost to a narrower code is
//! bought back more cheaply by reading more candidates than by scanning more bytes.

pub const DEFAULT_BITS: usize = 384;

/// Slots below this are scanned on the calling thread. Dispatching to a pool and joining costs
/// tens of microseconds; a single-threaded scan of 16k codes at 48 bytes is ~60 us, so below
/// roughly this size the dispatch is most of the query.
pub const MIN_PARALLEL_SLOTS: usize = 16384;

/// Codes per work item. Threads pull chunks off an atomic counter rather than taking an equal
/// share each, because this is a heterogeneous CPU: an equal split puts identical work on
/// performance and efficiency cores and the join waits for whichever landed on the slow one.
/// Measured, that turned a 4-thread scan into something slower than the 8-thread one. Chunks
/// small enough to rebalance and large enough that the atomic disappears fix it.
pub const CHUNK_SLOTS: usize = 32768;

pub const Error = error{
    /// `search` was given nowhere to put its results.
    EmptyBuffer,
};

/// What a saved codes file has to match to be worth loading: the store it was built from, and
/// the configuration it was built under.
///
/// **Counters alone cannot prove freshness**, and it is worth being explicit about why rather
/// than discovering it later. `replaceVectors` removes a document's rows and then puts the
/// replacements into the slots it just freed, so a re-embed of the same length leaves
/// `vec_n`, `slot_n` and the file size all exactly as they were while every code is wrong.
/// Detecting that would need either a mutation counter persisted inside the store -- whose
/// header is deliberately immutable -- or a rescan of every trailer, which is the cost the
/// file exists to avoid.
///
/// So the stamp is not the freshness mechanism. `load` deletes the file it read, which makes a
/// saved file valid only while nobody holds it open, and `save` writes a new one at shutdown.
/// The stamp's job is narrower and still worth doing: catch a file belonging to a *different*
/// store, or written by a build with a different code width or vector type.
pub const Stamp = struct {
    /// The store's byte length when the codes were written.
    store_bytes: u64,
    /// The store's slot high-water mark.
    slot_n: u64,
    /// The store's live vector count.
    vec_n: u64,

    fn eql(a: Stamp, b: Stamp) bool {
        return a.store_bytes == b.store_bytes and a.slot_n == b.slot_n and a.vec_n == b.vec_n;
    }
};

const MAGIC: [8]u8 = "DVECODES".*;
const FMT_V: u8 = 1;

/// Fixed-size and written first. Everything after it is two flat arrays: the liveness bits,
/// then the codes.
const FileHeader = extern struct {
    magic: [8]u8,
    fmt_v: u8,
    big_endian: u8,
    _pad: [6]u8 = @splat(0),
    code_bits: u64,
    vec_sz: u64,
    vec_elem_bits: u64,
    slot_n: u64,
    live_n: u64,
    store_bytes: u64,
    store_slot_n: u64,
    store_vec_n: u64,
};

pub fn Codes(comptime vec_sz: usize, comptime vec_type: type, comptime code_bits: usize) type {
    comptime {
        // A code is addressed as u64 words, and the scan reads it as 16-byte lanes with a
        // u64 tail, so any multiple of 64 works.
        assert(code_bits > 0);
        assert(code_bits % 64 == 0);
        assert(code_bits <= vec_sz);
    }

    return struct {
        const Self = @This();

        pub const CODE_BITS = code_bits;
        pub const CODE_BYTES = code_bits / 8;
        pub const CODE_WORDS = CODE_BYTES / 8;
        /// 16-byte NEON lanes, plus however many whole u64s are left over.
        const LANES = CODE_BYTES / 16;
        const TAIL_WORDS = (CODE_BYTES % 16) / 8;

        /// Matches `vstore`'s array form. A `@Vector(768, f32)` is padded to 4096 bytes, so
        /// anything crossing an API boundary is the bare array.
        pub const Array = [vec_sz]vec_type;
        pub const Word = [CODE_WORDS]u64;

        pub const Candidate = struct {
            slot: u32,
            /// Hamming distance in bits, 0..CODE_BITS.
            dist: u16,
        };

        pub const Opts = struct {
            /// Threads used for a scan. Null lets the pool size itself from the CPU count.
            /// Measured on a 4-performance + 6-efficiency core machine, four threads saturate
            /// memory bandwidth and everything above that is wasted -- but where that line
            /// sits is a property of the machine, so this is a knob rather than a constant.
            threads: ?usize = null,
            /// Slots to size the array for up front, avoiding the growth below.
            capacity: usize = 0,
        };

        allocator: std.mem.Allocator,
        /// `slot_n * CODE_WORDS` u64s, code `s` at `words[s * CODE_WORDS ..][0..CODE_WORDS]`.
        words: []u64 = &.{},
        /// One bit per slot. A dead slot has to be skipped explicitly rather than encoded as a
        /// sentinel, because there is no unused code: all-zero is what a vector whose first
        /// `code_bits` components are all negative encodes to, and it is as valid as any other.
        /// One bit per slot is 4.4 MB at 35M against 1.68 GB of codes, and it is a separate,
        /// cache-resident stream, so the check costs nothing the scan can measure.
        live: []u64 = &.{},
        /// High-water mark, matching the store's. Only grows.
        slot_n: usize = 0,
        live_n: usize = 0,
        pool: ?*std.Thread.Pool = null,

        // ************************************************************************* Lifecycle
        pub fn init(allocator: std.mem.Allocator, opts: Opts) !Self {
            var self = Self{ .allocator = allocator };
            errdefer self.deinit();
            if (opts.capacity > 0) try self.reserve(opts.capacity);

            const pool = try allocator.create(std.Thread.Pool);
            errdefer allocator.destroy(pool);
            try pool.init(.{ .allocator = allocator, .n_jobs = opts.threads });
            self.pool = pool;
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.pool) |pool| {
                pool.deinit();
                self.allocator.destroy(pool);
                self.pool = null;
            }
            self.allocator.free(self.words);
            self.allocator.free(self.live);
            self.words = &.{};
            self.live = &.{};
        }

        fn capacity(self: *const Self) usize {
            return if (CODE_WORDS == 0) 0 else self.words.len / CODE_WORDS;
        }

        /// Sizes the array for at least `slot_n` slots. Worth calling before a bulk build:
        /// growth is geometric, so the last reallocation of a 1.68 GB array copies all of it.
        pub fn reserve(self: *Self, want: usize) !void {
            if (want <= self.capacity()) return;
            var cap = @max(self.capacity(), 64);
            while (cap < want) cap *= 2;

            self.words = try self.allocator.realloc(self.words, cap * CODE_WORDS);
            const live_words = (cap + 63) / 64;
            const old_live = self.live.len;
            self.live = try self.allocator.realloc(self.live, live_words);
            // Only the liveness bits need initializing. Code bytes for a slot nothing has
            // written are never read: the scan skips any slot whose bit is clear.
            @memset(self.live[old_live..], 0);
        }

        // *************************************************************************** Encoding
        /// The sign of each of the first `code_bits` components, bit `i` from component `i`.
        ///
        /// `> 0` rather than `>= 0`: a component of exactly zero is arbitrary either way, and
        /// the comparison lowers to one SIMD op per 64 lanes whichever it is.
        pub fn encode(vec: *const Array, out: *Word) void {
            const Zero: @Vector(64, vec_type) = @splat(0);
            inline for (0..CODE_WORDS) |w| {
                const lane: @Vector(64, vec_type) = vec[w * 64 ..][0..64].*;
                // A `@Vector(64, bool)` is bit-packed, lane i at bit i, so the comparison
                // mask *is* the code word.
                out[w] = @bitCast(lane > Zero);
            }
        }

        pub fn hamming(a: *const Word, b: *const Word) u32 {
            const ab: *const [CODE_BYTES]u8 = @ptrCast(a);
            const bb: *const [CODE_BYTES]u8 = @ptrCast(b);
            var d: u32 = 0;
            if (LANES > 0) {
                // Sixteen bytes per `cnt` with no move to and from a general register, which
                // is what a per-u64 `@popCount` costs on AArch64. Per-lane counts cap at 8, so
                // `LANES * 8` fits in u8 for any code width that matters -- but the 16-lane
                // reduction would not, so it widens first.
                comptime assert(LANES * 8 <= std.math.maxInt(u8));
                var acc: @Vector(16, u8) = @splat(0);
                inline for (0..LANES) |l| {
                    const av: @Vector(16, u8) = ab[l * 16 ..][0..16].*;
                    const bv: @Vector(16, u8) = bb[l * 16 ..][0..16].*;
                    acc +%= @popCount(av ^ bv);
                }
                const wide: @Vector(16, u16) = acc;
                d = @reduce(.Add, wide);
            }
            inline for (0..TAIL_WORDS) |t| {
                d += @popCount(a[LANES * 2 + t] ^ b[LANES * 2 + t]);
            }
            return d;
        }

        fn codeAt(self: *const Self, slot: usize) *const Word {
            return @ptrCast(self.words[slot * CODE_WORDS ..][0..CODE_WORDS]);
        }

        // *************************************************************************** Mutation
        pub fn isLive(self: *const Self, slot: usize) bool {
            if (slot >= self.slot_n) return false;
            return self.live[slot / 64] >> @intCast(slot % 64) & 1 == 1;
        }

        /// Encodes `vec` into `slot`, which may be new, or may be a slot the store has just
        /// handed back out after a remove. Overwriting is the whole point: the store reuses
        /// slots, so this array stays as dense as the store's slot space and never needs
        /// compacting.
        pub fn put(self: *Self, slot: usize, vec: *const Array) !void {
            try self.reserve(slot + 1);
            if (slot >= self.slot_n) self.slot_n = slot + 1;

            const out: *Word = @ptrCast(self.words[slot * CODE_WORDS ..][0..CODE_WORDS]);
            encode(vec, out);
            if (!self.isLive(slot)) self.live_n += 1;
            self.live[slot / 64] |= @as(u64, 1) << @intCast(slot % 64);
        }

        pub fn rm(self: *Self, slot: usize) void {
            if (!self.isLive(slot)) return;
            self.live[slot / 64] &= ~(@as(u64, 1) << @intCast(slot % 64));
            self.live_n -= 1;
            // The code bytes are left as they are. Nothing reads a dead slot, and clearing
            // them would touch a page for no reason.
        }

        pub fn len(self: *const Self) usize {
            return self.live_n;
        }
        pub fn slotCount(self: *const Self) usize {
            return self.slot_n;
        }
        /// Resident bytes, which is the number this whole design is sized against.
        pub fn bytes(self: *const Self) usize {
            return self.words.len * 8 + self.live.len * 8;
        }

        // ************************************************************************ Persistence
        /// Byte offsets of the two arrays. Both are `u64`-aligned by construction, so a load
        /// reads straight into them with no copying or byte-shuffling.
        fn liveOff() u64 {
            return @sizeOf(FileHeader);
        }
        fn codesOff(live_words: usize) u64 {
            return liveOff() + live_words * 8;
        }

        /// Writes the index to `path` in `dir`, replacing whatever was there.
        ///
        /// No checksum over the payload, deliberately. At 1.68 GB a crc32 costs about a
        /// second, which is several times the read it is protecting, and the failure it would
        /// catch is uniquely benign here: every bit pattern is a valid code, so a corrupted
        /// one cannot crash anything or return wrong data -- stage two re-reads the real
        /// vector and scores it exactly. A flipped bit costs a little recall and nothing else.
        /// A *truncated* file is caught, by length.
        pub fn save(self: *const Self, dir: std.fs.Dir, path: []const u8, stamp: Stamp) !void {
            const live_words = (self.slot_n + 63) / 64;
            const code_words = self.slot_n * CODE_WORDS;

            const file = try pfile.File.openAt(@intCast(dir.fd), path, .{ .truncate = true });
            defer file.close();

            const h = FileHeader{
                .magic = MAGIC,
                .fmt_v = FMT_V,
                .big_endian = @intFromBool(native_endian == .big),
                .code_bits = code_bits,
                .vec_sz = vec_sz,
                .vec_elem_bits = @bitSizeOf(vec_type),
                .slot_n = self.slot_n,
                .live_n = self.live_n,
                .store_bytes = stamp.store_bytes,
                .store_slot_n = stamp.slot_n,
                .store_vec_n = stamp.vec_n,
            };
            try file.writeAt(std.mem.asBytes(&h), 0);
            try file.writeAt(std.mem.sliceAsBytes(self.live[0..live_words]), liveOff());
            try file.writeAt(
                std.mem.sliceAsBytes(self.words[0..code_words]),
                codesOff(live_words),
            );
            try file.sync();
        }

        /// Loads an index written by `save`, or returns null if there is nothing usable --
        /// absent, a different format, a different configuration, or built from a different
        /// store. Null is the normal path, not an error: the caller rebuilds.
        ///
        /// **A successful load deletes the file**, and that is the freshness mechanism rather
        /// than an optimisation. See `Stamp` for why counters cannot do the job. Removing the
        /// file means a saved index is only ever valid while no process holds it, so a crash
        /// -- or any exit that does not reach `save` -- leaves nothing to load and the next
        /// open rebuilds. The cost of being wrong here is silently missing search results, and
        /// the cost of being conservative is one rebuild, so it is not a close call.
        pub fn load(
            allocator: std.mem.Allocator,
            dir: std.fs.Dir,
            path: []const u8,
            stamp: Stamp,
            opts: Opts,
        ) !?Self {
            const file = pfile.File.openAt(@intCast(dir.fd), path, .{ .create = false }) catch
                return null;
            var keep_file = true;
            defer {
                file.close();
                if (!keep_file) dir.deleteFile(path) catch {};
            }

            var h: FileHeader = undefined;
            const got = file.readAt(std.mem.asBytes(&h), 0) catch return null;
            if (got != @sizeOf(FileHeader)) return null;

            const ok = std.mem.eql(u8, &h.magic, &MAGIC) and
                h.fmt_v == FMT_V and
                h.big_endian == @intFromBool(native_endian == .big) and
                h.code_bits == code_bits and
                h.vec_sz == vec_sz and
                h.vec_elem_bits == @bitSizeOf(vec_type) and
                Stamp.eql(.{
                    .store_bytes = h.store_bytes,
                    .slot_n = h.store_slot_n,
                    .vec_n = h.store_vec_n,
                }, stamp);
            if (!ok) {
                // Stale or foreign. Drop it rather than leave it to be re-rejected forever.
                keep_file = false;
                return null;
            }

            const slot_n: usize = @intCast(h.slot_n);
            const live_words = (slot_n + 63) / 64;
            const code_words = slot_n * CODE_WORDS;
            const want = codesOff(live_words) + code_words * 8;
            if ((file.size() catch return null) != want) {
                keep_file = false; // truncated: caught by length, which is what length is for
                return null;
            }

            var self = try Self.init(allocator, .{ .capacity = slot_n, .threads = opts.threads });
            errdefer self.deinit();

            if (slot_n > 0) {
                const lv = self.live[0..live_words];
                if (try file.readAt(std.mem.sliceAsBytes(lv), liveOff()) != live_words * 8) {
                    return null;
                }
                const cw = self.words[0..code_words];
                if (try file.readAt(
                    std.mem.sliceAsBytes(cw),
                    codesOff(live_words),
                ) != code_words * 8) return null;
            }
            self.slot_n = slot_n;
            self.live_n = @intCast(h.live_n);

            keep_file = false;
            return self;
        }

        // ***************************************************************************** Search
        /// Stage one. Fills `out` with the closest live slots by Hamming distance, nearest
        /// first, and returns how many. `out.len` is K -- the candidate count, which is also
        /// the number of disk reads stage two will do, which is what makes it the cost knob.
        pub fn search(self: *Self, query: *const Array, out: []Candidate) !usize {
            if (out.len == 0) return Error.EmptyBuffer;
            if (self.slot_n == 0) return 0;

            var q: Word = undefined;
            encode(query, &q);

            const threads = if (self.pool) |p| p.threads.len else 1;
            if (threads <= 1 or self.slot_n < MIN_PARALLEL_SLOTS) {
                var top = TopK.init(out);
                self.scanRange(&q, 0, self.slot_n, &top);
                return top.n;
            }
            return self.searchParallel(&q, out, threads);
        }

        fn searchParallel(self: *Self, q: *const Word, out: []Candidate, threads: usize) !usize {
            const shard = try self.allocator.alloc(Candidate, threads * out.len);
            defer self.allocator.free(shard);
            const tops = try self.allocator.alloc(TopK, threads);
            defer self.allocator.free(tops);

            var next = std.atomic.Value(usize).init(0);
            var wg: std.Thread.WaitGroup = .{};
            for (tops, 0..) |*top, t| {
                top.* = TopK.init(shard[t * out.len ..][0..out.len]);
                self.pool.?.spawnWg(&wg, workerRun, .{ self, q, &next, top });
            }
            self.pool.?.waitAndWork(&wg);

            // A real k-way merge, because the shards are already sorted and the obvious
            // alternative is quadratic. Pushing every shard entry into one more `TopK` costs
            // O(threads * K^2): the first shard appends cheaply, but every shard after it
            // interleaves, and each insertion shifts about half the array. Measured at K=4000
            // that was 62 ms a query against 0.5 ms at K=100 -- superlinear in K, and none of
            // it disk. Taking the best head `threads` times per output slot is O(K * threads),
            // and `threads` is single digits.
            const head = try self.allocator.alloc(usize, threads);
            defer self.allocator.free(head);
            @memset(head, 0);

            var n: usize = 0;
            while (n < out.len) : (n += 1) {
                var best: usize = 0;
                var best_key: u64 = std.math.maxInt(u64);
                for (tops, head, 0..) |top, h, t| {
                    if (h >= top.n) continue;
                    const key = keyOfCand(top.buf[h]);
                    if (key < best_key) {
                        best_key = key;
                        best = t;
                    }
                }
                if (best_key == std.math.maxInt(u64)) break; // every shard drained
                out[n] = tops[best].buf[head[best]];
                head[best] += 1;
            }
            return n;
        }

        fn workerRun(self: *Self, q: *const Word, next: *std.atomic.Value(usize), top: *TopK) void {
            while (true) {
                const chunk = next.fetchAdd(1, .monotonic);
                const lo = chunk * CHUNK_SLOTS;
                if (lo >= self.slot_n) break;
                const hi = @min(lo + CHUNK_SLOTS, self.slot_n);
                self.scanRange(q, lo, hi, top);
            }
        }

        fn scanRange(self: *const Self, q: *const Word, lo: usize, hi: usize, top: *TopK) void {
            var slot = lo;
            while (slot < hi) : (slot += 1) {
                if (!self.isLive(slot)) continue;
                const d = hamming(q, self.codeAt(slot));
                // One u64 comparison rejects almost everything, and it is the *whole* ordering
                // including the tie-break -- see `keyOf`. After the first few hundred
                // candidates `worst_key` has converged and a push happens about K*ln(N/K)
                // times in a scan of N.
                const key = keyOf(d, @intCast(slot));
                if (key < top.worst_key) top.push(key);
            }
        }

        /// Bounded top-K over a caller-owned buffer, kept sorted nearest-first.
        ///
        /// **Insertion, not a heap, and that caps the useful K at around 1,000.** A push costs
        /// O(K) to shift, and the number of pushes in a scan of N is about K*ln(N/K), so the
        /// selection is O(K^2 log) overall. Below ~500 it is invisible -- `worst_key`
        /// converges in the first few hundred candidates and almost everything after that is
        /// rejected by one comparison. Above it, it takes over: measured on a pure in-memory
        /// scan of 5M codes with no disk involved, K=500 is 10 ms and K=4,000 is 150 ms.
        ///
        /// That is fine for the K this is used at -- 500 by default, which reaches 0.99 recall
        /// (`experiments/results/binrecall.md`) -- and it is a trap for anyone who raises
        /// `candidates` expecting a linear cost. A binary heap would make a push O(log K) and
        /// remove the ceiling; it is not written because nothing needs K that large yet, and
        /// because a heap gives up the sorted result the caller currently gets for free.
        ///
        /// **Ordered by (distance, slot), not distance alone**, and that second key is not
        /// tidiness. Hamming distances are small integers over a large corpus, so ties are
        /// everywhere -- a corpus with repeated text puts hundreds of slots at distance 0 from
        /// one query. Breaking those by whichever thread happened to scan its chunk first
        /// means the same query returns different results on consecutive runs, which is a bug
        /// a user sees. Ordering by slot makes the answer the global top-K by (distance, slot)
        /// no matter how the work was divided.
        const TopK = struct {
            buf: []Candidate,
            n: usize = 0,
            worst_key: u64 = std.math.maxInt(u64),

            fn init(buf: []Candidate) TopK {
                return .{ .buf = buf };
            }

            fn push(self: *TopK, key: u64) void {
                const k = self.buf.len;
                var j = @min(self.n, k - 1);
                while (j > 0 and key < keyOfCand(self.buf[j - 1])) : (j -= 1) {
                    self.buf[j] = self.buf[j - 1];
                }
                self.buf[j] = .{ .slot = slotOfKey(key), .dist = @intCast(key >> 32) };
                if (self.n < k) self.n += 1;
                if (self.n == k) self.worst_key = keyOfCand(self.buf[k - 1]);
            }
        };

        /// Distance in the high half, slot in the low half, so one unsigned compare orders
        /// candidates by distance and breaks ties by slot. That is what keeps the hot path at
        /// a single comparison while still making the answer independent of how the scan was
        /// divided across threads -- ordering by distance alone costs an extra compare per
        /// code, which is ~50% of a cache-resident scan.
        fn keyOf(d: u32, slot: u32) u64 {
            return @as(u64, d) << 32 | slot;
        }
        fn keyOfCand(c: Candidate) u64 {
            return keyOf(c.dist, c.slot);
        }
        fn slotOfKey(key: u64) u32 {
            return @truncate(key);
        }
    };
}

const std = @import("std");
const assert = std.debug.assert;
const native_endian = @import("builtin").cpu.arch.endian();
const pfile = @import("pfile.zig");

// ****************************************************************************************** Tests
const testing = std.testing;
const talloc = testing.allocator;
const expectEqual = testing.expectEqual;

/// One u64 of code, which is the narrowest a code can be. The test patterns below only ever
/// set bits in the low byte, so distances stay countable by hand.
const Tiny = Codes(64, f32, 64);
/// 256 components with a 128-bit code: exercises both truncation and the 16-byte lane path.
const Wide = Codes(256, f32, 128);

fn tinyVec(pattern: u64) [64]f32 {
    var v: [64]f32 = undefined;
    for (&v, 0..) |*x, i| x.* = if (pattern >> @intCast(i) & 1 == 1) 1.0 else -1.0;
    return v;
}

test "encode: bit i is the sign of component i" {
    var v: [64]f32 = undefined;
    for (&v, 0..) |*x, i| x.* = if (i % 3 == 0) 0.5 else -0.5;

    var code: Tiny.Word = undefined;
    Tiny.encode(&v, &code);

    for (0..64) |i| {
        const bit = code[0] >> @intCast(i) & 1;
        try expectEqual(@as(u64, if (i % 3 == 0) 1 else 0), bit);
    }
}

test "encode: a vector and its negation differ in every bit" {
    var v: [64]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&v) |*x| x.* = prng.random().floatNorm(f32);
    var neg: [64]f32 = undefined;
    for (&neg, v) |*n, x| n.* = -x;

    var a: Tiny.Word = undefined;
    var b: Tiny.Word = undefined;
    Tiny.encode(&v, &a);
    Tiny.encode(&neg, &b);
    try expectEqual(@as(u32, 64), Tiny.hamming(&a, &b));
    try expectEqual(@as(u32, 0), Tiny.hamming(&a, &a));
}

test "encode: only the first code_bits components are read" {
    // 256 components, 128-bit code. Changing a component past the cutoff must not change the
    // code -- this is what makes a narrow code a truncation rather than a mix.
    var v: [256]f32 = @splat(-1.0);
    var a: Wide.Word = undefined;
    Wide.encode(&v, &a);

    v[200] = 1.0;
    var b: Wide.Word = undefined;
    Wide.encode(&v, &b);
    try expectEqual(@as(u32, 0), Wide.hamming(&a, &b));

    v[7] = 1.0;
    Wide.encode(&v, &b);
    try expectEqual(@as(u32, 1), Wide.hamming(&a, &b));
}

test "hamming: 16-byte lanes and the u64 tail agree" {
    // 192 bits is 24 bytes: one 16-byte lane plus one u64 of tail, which is the path a width
    // that is not a multiple of 128 takes and the one most likely to be wrong.
    const C = Codes(192, f32, 192);
    var v: [192]f32 = @splat(-1.0);
    var a: C.Word = undefined;
    C.encode(&v, &a);

    // Flip one component in the lane region and one in the tail.
    v[3] = 1.0; // bit 3, first lane
    v[170] = 1.0; // bit 170, inside the tail word
    var b: C.Word = undefined;
    C.encode(&v, &b);
    try expectEqual(@as(u32, 2), C.hamming(&a, &b));
}

test "put and rm track liveness and the slot high-water mark" {
    var c = try Tiny.init(talloc, .{});
    defer c.deinit();

    const v = tinyVec(0b1010);
    try expectEqual(@as(usize, 0), c.len());
    try expectEqual(@as(usize, 0), c.slotCount());

    try c.put(0, &v);
    try c.put(5, &v);
    try expectEqual(@as(usize, 2), c.len());
    // Slot 5 raises the high-water mark past the hole at 1..4, exactly as the store's does.
    try expectEqual(@as(usize, 6), c.slotCount());
    try testing.expect(c.isLive(0));
    try testing.expect(!c.isLive(3));
    try testing.expect(c.isLive(5));

    c.rm(0);
    try expectEqual(@as(usize, 1), c.len());
    try testing.expect(!c.isLive(0));
    // The mark does not fall: it bounds the scan, and a freed slot is still scannable space.
    try expectEqual(@as(usize, 6), c.slotCount());

    // Removing twice is not an error, matching how the store tolerates a stale id.
    c.rm(0);
    try expectEqual(@as(usize, 1), c.len());
}

test "put overwrites a reused slot rather than appending" {
    var c = try Tiny.init(talloc, .{});
    defer c.deinit();

    const a = tinyVec(0b00000000);
    const b = tinyVec(0b11111111);
    try c.put(3, &a);
    c.rm(3);
    try c.put(3, &b);

    try expectEqual(@as(usize, 1), c.len());
    try expectEqual(@as(usize, 4), c.slotCount());

    // The slot now answers for b, not a: a query equal to b finds it at distance 0.
    var buf: [1]Tiny.Candidate = undefined;
    const n = try c.search(&b, &buf);
    try expectEqual(@as(usize, 1), n);
    try expectEqual(@as(u32, 3), buf[0].slot);
    try expectEqual(@as(u16, 0), buf[0].dist);
}

test "search: exact match first, results sorted, dead slots excluded" {
    var c = try Tiny.init(talloc, .{});
    defer c.deinit();

    // Distances from 0b00000000: slot 0 is 0 bits away, slot 1 is 1, slot 2 is 2, slot 3 is 3.
    try c.put(0, &tinyVec(0b00000000));
    try c.put(1, &tinyVec(0b00000001));
    try c.put(2, &tinyVec(0b00000011));
    try c.put(3, &tinyVec(0b00000111));

    const q = tinyVec(0b00000000);
    var buf: [4]Tiny.Candidate = undefined;
    var n = try c.search(&q, &buf);
    try expectEqual(@as(usize, 4), n);
    for (buf[0..n], 0..) |cand, i| {
        try expectEqual(@as(u32, @intCast(i)), cand.slot);
        try expectEqual(@as(u16, @intCast(i)), cand.dist);
    }

    // A removed slot must not come back as a candidate, even though its code bytes are still
    // sitting there and are the best match in the array.
    c.rm(0);
    n = try c.search(&q, &buf);
    try expectEqual(@as(usize, 3), n);
    try expectEqual(@as(u32, 1), buf[0].slot);
}

test "search: K bounds the result, keeping the nearest" {
    var c = try Tiny.init(talloc, .{});
    defer c.deinit();
    for (0..8) |i| {
        const bits = (@as(u64, 1) << @intCast(i)) - 1; // i bits set -> distance i
        try c.put(i, &tinyVec(bits));
    }

    const q = tinyVec(0);
    var buf: [3]Tiny.Candidate = undefined;
    const n = try c.search(&q, &buf);
    try expectEqual(@as(usize, 3), n);
    try expectEqual(@as(u16, 0), buf[0].dist);
    try expectEqual(@as(u16, 1), buf[1].dist);
    try expectEqual(@as(u16, 2), buf[2].dist);
}

test "search: an empty buffer is an error, an empty index is not" {
    var c = try Tiny.init(talloc, .{});
    defer c.deinit();
    const q = tinyVec(0);

    var none: [0]Tiny.Candidate = undefined;
    try testing.expectError(Error.EmptyBuffer, c.search(&q, &none));

    var buf: [4]Tiny.Candidate = undefined;
    try expectEqual(@as(usize, 0), try c.search(&q, &buf));

    // Every slot removed is also zero results, not a stale one.
    try c.put(0, &q);
    c.rm(0);
    try expectEqual(@as(usize, 0), try c.search(&q, &buf));
}

// Big enough to cross MIN_PARALLEL_SLOTS so the threaded path actually runs, and required to
// agree with the single-threaded one exactly. The work-stealing split and the shard merge are
// the two places a parallel scan can silently lose a candidate, and neither is exercised by
// any test above.
test "search: threaded and single-threaded agree exactly" {
    const N = MIN_PARALLEL_SLOTS + CHUNK_SLOTS + 1234; // spans several chunks, ragged tail
    const C = Tiny;

    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();

    var par = try C.init(talloc, .{ .threads = 4, .capacity = N });
    defer par.deinit();
    var seq = try C.init(talloc, .{ .threads = 1, .capacity = N });
    defer seq.deinit();

    var v: [64]f32 = undefined;
    for (0..N) |slot| {
        for (&v) |*x| x.* = rng.floatNorm(f32);
        try par.put(slot, &v);
        try seq.put(slot, &v);
        // Scatter holes so the liveness check is exercised across chunk boundaries too.
        if (slot % 17 == 0) {
            par.rm(slot);
            seq.rm(slot);
        }
    }
    try expectEqual(seq.len(), par.len());

    var qbuf: [64]f32 = undefined;
    var a: [64]C.Candidate = undefined;
    var b: [64]C.Candidate = undefined;
    for (0..8) |_| {
        for (&qbuf) |*x| x.* = rng.floatNorm(f32);
        const na = try par.search(&qbuf, &a);
        const nb = try seq.search(&qbuf, &b);
        try expectEqual(nb, na);
        // Distances must match exactly. Slots may differ only where distances tie, so compare
        // the distance sequence and require every returned slot to actually hold that code.
        for (a[0..na], b[0..nb]) |ca, cb| {
            try expectEqual(cb.dist, ca.dist);
            try testing.expect(par.isLive(ca.slot));
        }
    }
}

test "search: parallel merge keeps the true nearest across shard boundaries" {
    // One deliberately perfect match placed in the last chunk, where a merge bug that favours
    // the first shard would lose it.
    const N = MIN_PARALLEL_SLOTS + CHUNK_SLOTS + 500;
    const C = Tiny;

    var c = try C.init(talloc, .{ .threads = 4, .capacity = N });
    defer c.deinit();

    var v: [64]f32 = @splat(1.0);
    for (0..N) |slot| try c.put(slot, &v);

    var q: [64]f32 = @splat(-1.0); // distance 64 from every filler
    try c.put(N - 1, &q); // ...except this one, at distance 0

    var buf: [4]C.Candidate = undefined;
    const n = try c.search(&q, &buf);
    try expectEqual(@as(usize, 4), n);
    try expectEqual(@as(u32, @intCast(N - 1)), buf[0].slot);
    try expectEqual(@as(u16, 0), buf[0].dist);
    try expectEqual(@as(u16, 64), buf[1].dist);
}

test "reserve: growth preserves codes and liveness" {
    var c = try Tiny.init(talloc, .{});
    defer c.deinit();

    // Put sparsely so the array reallocates several times, then check nothing moved.
    const slots = [_]usize{ 0, 1, 63, 64, 65, 200, 1000 };
    for (slots, 0..) |slot, i| try c.put(slot, &tinyVec(@as(u64, i) + 1));
    try expectEqual(@as(usize, slots.len), c.len());
    try expectEqual(@as(usize, 1001), c.slotCount());

    for (slots, 0..) |slot, i| {
        try testing.expect(c.isLive(slot));
        var buf: [1]Tiny.Candidate = undefined;
        const n = try c.search(&tinyVec(@as(u64, i) + 1), &buf);
        try expectEqual(@as(usize, 1), n);
        try expectEqual(@as(u16, 0), buf[0].dist);
    }
    // Nothing between the sparse slots came back alive.
    try testing.expect(!c.isLive(2));
    try testing.expect(!c.isLive(999));
}

test "bytes: the resident cost is the code array plus one bit per slot" {
    const C = Codes(768, f32, 384);
    var c = try C.init(talloc, .{ .capacity = 1024 });
    defer c.deinit();
    // 1024 slots x 48 bytes, plus 1024 bits of liveness.
    try expectEqual(@as(usize, 1024 * 48 + 1024 / 8), c.bytes());
}

test "the production configuration is 48 bytes a vector" {
    try expectEqual(@as(usize, 384), Prod.CODE_BITS);
    try expectEqual(@as(usize, 48), Prod.CODE_BYTES);
    try expectEqual(@as(usize, 6), Prod.CODE_WORDS);
    // 35M vectors is the Simple Wikipedia corpus as it is indexed today.
    try expectEqual(@as(usize, 1_680_000_000), 35_000_000 * Prod.CODE_BYTES);
}

// A corpus full of exact duplicates, which is what the real one is: the Simple Wikipedia
// chunks include hundreds of copies of "Related pages" and friends, so a typical query sits
// at distance 0 from ~500 slots. Every test above used distinct random codes, so the tie path
// -- where the shard merge has to choose among equals -- was never exercised.
test "search: recall cannot fall as K grows, even when everything ties" {
    const N = MIN_PARALLEL_SLOTS + CHUNK_SLOTS + 777;
    var c = try Tiny.init(talloc, .{ .threads = 4, .capacity = N });
    defer c.deinit();

    // Two thirds of the corpus is an exact duplicate of the query; the rest is far away.
    const q = tinyVec(0);
    const far = tinyVec(std.math.maxInt(u64));
    for (0..N) |slot| {
        try c.put(slot, if (slot % 3 == 0) &far else &q);
    }

    var buf: [4096]Tiny.Candidate = undefined;
    var prev: usize = 0;
    for ([_]usize{ 10, 50, 100, 500, 1000, 2000, 4096 }) |k| {
        const found = try c.search(&q, buf[0..k]);
        try expectEqual(k, found);
        // Every candidate must be one of the duplicates until they run out: a merge that
        // drops ties would let a distance-64 filler in while distance-0 slots went unreturned.
        var zeros: usize = 0;
        for (buf[0..found]) |cand| {
            if (cand.dist == 0) zeros += 1;
        }
        try testing.expect(zeros >= prev);
        try expectEqual(@min(k, N - (N + 2) / 3), zeros);
        prev = zeros;
    }
}

test "search: a tied corpus returns each slot at most once" {
    const N = MIN_PARALLEL_SLOTS + CHUNK_SLOTS + 33;
    var c = try Tiny.init(talloc, .{ .threads = 4, .capacity = N });
    defer c.deinit();
    const q = tinyVec(0);
    for (0..N) |slot| try c.put(slot, &q); // every code identical

    var buf: [1000]Tiny.Candidate = undefined;
    const found = try c.search(&q, &buf);
    try expectEqual(@as(usize, 1000), found);

    var seen = std.AutoHashMap(u32, void).init(talloc);
    defer seen.deinit();
    for (buf[0..found]) |cand| {
        try expectEqual(@as(u16, 0), cand.dist);
        const gop = try seen.getOrPut(cand.slot);
        try testing.expect(!gop.found_existing); // a duplicated slot is a merge bug
    }
}

// The property the two-stage design actually depends on: a larger candidate budget can only
// help. If the top-500 is not a distance-wise extension of the top-100, then raising K can
// lose a true neighbour, and the whole "spend the code's savings on a bigger K" argument in
// binrecall.md stops holding.
test "search: a larger K extends the smaller one rather than replacing it" {
    const N = MIN_PARALLEL_SLOTS + CHUNK_SLOTS + 555;
    var c = try Tiny.init(talloc, .{ .threads = 4, .capacity = N });
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(4242);
    const rng = prng.random();
    var v: [64]f32 = undefined;
    for (0..N) |slot| {
        for (&v) |*x| x.* = rng.floatNorm(f32);
        try c.put(slot, &v);
    }

    var small: [100]Tiny.Candidate = undefined;
    var large: [1000]Tiny.Candidate = undefined;
    for (0..6) |_| {
        for (&v) |*x| x.* = rng.floatNorm(f32);
        const ns = try c.search(&v, &small);
        const nl = try c.search(&v, &large);
        try expectEqual(@as(usize, 100), ns);
        try expectEqual(@as(usize, 1000), nl);

        // Distances are sorted in both, so the smaller run's distances must be exactly the
        // large run's first 100.
        for (small[0..ns], large[0..ns]) |a, b| try expectEqual(b.dist, a.dist);

        // And every slot the small run found must still be present in the large one.
        for (small[0..ns]) |a| {
            var present = false;
            for (large[0..nl]) |b| {
                if (b.slot == a.slot) {
                    present = true;
                    break;
                }
            }
            try testing.expect(present);
        }
    }
}

// Determinism is a product property, not an implementation detail: a user who runs the same
// search twice must see the same results. With work-stealing across a heterogeneous CPU,
// nothing about which thread scans which chunk is stable, so ties have to be broken by
// something intrinsic to the data. These two tests are what hold that down.
test "search: a tied corpus returns the same answer every time" {
    const N = MIN_PARALLEL_SLOTS + CHUNK_SLOTS + 91;
    var c = try Tiny.init(talloc, .{ .threads = 4, .capacity = N });
    defer c.deinit();

    const q = tinyVec(0);
    const far = tinyVec(std.math.maxInt(u64));
    for (0..N) |slot| try c.put(slot, if (slot % 5 == 0) &far else &q);

    var first: [256]Tiny.Candidate = undefined;
    var again: [256]Tiny.Candidate = undefined;
    const n0 = try c.search(&q, &first);
    for (0..12) |_| {
        const n = try c.search(&q, &again);
        try expectEqual(n0, n);
        try testing.expectEqualSlices(Tiny.Candidate, first[0..n0], again[0..n]);
    }
    // And the order is the documented one: distance, then slot.
    for (first[0 .. n0 - 1], first[1..n0]) |a, b| {
        try testing.expect(a.dist < b.dist or (a.dist == b.dist and a.slot < b.slot));
    }
}

test "search: threading does not change the answer, ties included" {
    const N = MIN_PARALLEL_SLOTS + CHUNK_SLOTS + 404;
    var par = try Tiny.init(talloc, .{ .threads = 4, .capacity = N });
    defer par.deinit();
    var seq = try Tiny.init(talloc, .{ .threads = 1, .capacity = N });
    defer seq.deinit();

    // Deliberately few distinct codes, so almost every comparison is a tie.
    for (0..N) |slot| {
        const v = tinyVec(@as(u64, slot % 7));
        try par.put(slot, &v);
        try seq.put(slot, &v);
    }

    var a: [300]Tiny.Candidate = undefined;
    var b: [300]Tiny.Candidate = undefined;
    for (0..7) |p| {
        const q = tinyVec(@as(u64, p));
        const na = try par.search(&q, &a);
        const nb = try seq.search(&q, &b);
        try expectEqual(nb, na);
        // Slot for slot, not just distance for distance.
        try testing.expectEqualSlices(Tiny.Candidate, b[0..nb], a[0..na]);
    }
}

// ****************************************************************************** Persistence
const Prod = Codes(768, f32, DEFAULT_BITS);

fn randVec(comptime N: usize, rng: std.Random, out: *[N]f32) void {
    for (out) |*x| x.* = rng.floatNorm(f32);
}

test "save then load reproduces the index exactly" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const stamp = Stamp{ .store_bytes = 4096 * 300, .slot_n = 300, .vec_n = 280 };

    var v: [768]f32 = undefined;
    var saved = try Prod.init(talloc, .{ .capacity = 300 });
    defer saved.deinit();
    for (0..300) |slot| {
        randVec(768, rng, &v);
        try saved.put(slot, &v);
        if (slot % 15 == 0) saved.rm(slot); // holes, so liveness has to survive the round trip
    }
    try saved.save(tmp.dir, "c.bin", stamp);

    var loaded = (try Prod.load(talloc, tmp.dir, "c.bin", stamp, .{})).?;
    defer loaded.deinit();

    try expectEqual(saved.slotCount(), loaded.slotCount());
    try expectEqual(saved.len(), loaded.len());
    for (0..300) |slot| try expectEqual(saved.isLive(slot), loaded.isLive(slot));
    try testing.expectEqualSlices(u64, saved.words[0 .. 300 * Prod.CODE_WORDS], loaded.words[0 .. 300 * Prod.CODE_WORDS]);

    // And it answers the same, which is the property that actually matters.
    var qv: [768]f32 = undefined;
    var a: [20]Prod.Candidate = undefined;
    var b: [20]Prod.Candidate = undefined;
    for (0..5) |_| {
        randVec(768, rng, &qv);
        const na = try saved.search(&qv, &a);
        const nb = try loaded.search(&qv, &b);
        try expectEqual(na, nb);
        try testing.expectEqualSlices(Prod.Candidate, a[0..na], b[0..nb]);
    }
}

test "load consumes the file, so a second open has nothing to read" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const stamp = Stamp{ .store_bytes = 4096, .slot_n = 1, .vec_n = 1 };

    var v: [768]f32 = @splat(0.5);
    var c = try Prod.init(talloc, .{ .capacity = 1 });
    defer c.deinit();
    try c.put(0, &v);
    try c.save(tmp.dir, "c.bin", stamp);

    var first = (try Prod.load(talloc, tmp.dir, "c.bin", stamp, .{})).?;
    first.deinit();

    // Gone. A saved index is only valid while nobody holds it, which is what stops a crashed
    // process leaving a stale one behind -- see the comment on `load`.
    try expectEqual(@as(?Prod, null), try Prod.load(talloc, tmp.dir, "c.bin", stamp, .{}));
    try testing.expectError(error.FileNotFound, tmp.dir.access("c.bin", .{}));
}

test "load refuses a file from a different store, and does not leave it behind" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const written = Stamp{ .store_bytes = 4096 * 10, .slot_n = 10, .vec_n = 10 };

    var v: [768]f32 = @splat(0.25);
    var c = try Prod.init(talloc, .{ .capacity = 10 });
    defer c.deinit();
    for (0..10) |slot| try c.put(slot, &v);

    // Each field on its own is enough to reject.
    for ([_]Stamp{
        .{ .store_bytes = 4096 * 11, .slot_n = 10, .vec_n = 10 },
        .{ .store_bytes = 4096 * 10, .slot_n = 11, .vec_n = 10 },
        .{ .store_bytes = 4096 * 10, .slot_n = 10, .vec_n = 9 },
    }) |wrong| {
        try c.save(tmp.dir, "c.bin", written);
        try expectEqual(@as(?Prod, null), try Prod.load(talloc, tmp.dir, "c.bin", wrong, .{}));
        try testing.expectError(error.FileNotFound, tmp.dir.access("c.bin", .{}));
    }
}

test "load refuses a file written for a different code width" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const stamp = Stamp{ .store_bytes = 4096 * 4, .slot_n = 4, .vec_n = 4 };

    const Narrow = Codes(768, f32, 128);
    var v: [768]f32 = @splat(1.0);
    var c = try Narrow.init(talloc, .{ .capacity = 4 });
    defer c.deinit();
    for (0..4) |slot| try c.put(slot, &v);
    try c.save(tmp.dir, "c.bin", stamp);

    // Same store, same vectors, a build that quantizes differently. Loading it would be
    // reading 48-byte codes out of 16-byte ones.
    try expectEqual(@as(?Prod, null), try Prod.load(talloc, tmp.dir, "c.bin", stamp, .{}));
}

test "load refuses a truncated file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const stamp = Stamp{ .store_bytes = 4096 * 50, .slot_n = 50, .vec_n = 50 };

    var prng = std.Random.DefaultPrng.init(3);
    var v: [768]f32 = undefined;
    var c = try Prod.init(talloc, .{ .capacity = 50 });
    defer c.deinit();
    for (0..50) |slot| {
        randVec(768, prng.random(), &v);
        try c.put(slot, &v);
    }
    try c.save(tmp.dir, "c.bin", stamp);

    // Lop off the last few codes, which is what a crash mid-write looks like. There is no
    // checksum over the payload -- see `save` for why -- so length is the whole defence.
    const f = try tmp.dir.openFile("c.bin", .{ .mode = .read_write });
    const full = (try f.stat()).size;
    try f.setEndPos(full - 200);
    f.close();

    try expectEqual(@as(?Prod, null), try Prod.load(talloc, tmp.dir, "c.bin", stamp, .{}));
}

test "load of a missing file is null, not an error" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const stamp = Stamp{ .store_bytes = 0, .slot_n = 0, .vec_n = 0 };
    try expectEqual(@as(?Prod, null), try Prod.load(talloc, tmp.dir, "nope.bin", stamp, .{}));
}

test "an empty index round-trips" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const stamp = Stamp{ .store_bytes = 4096, .slot_n = 0, .vec_n = 0 };

    var c = try Prod.init(talloc, .{});
    defer c.deinit();
    try c.save(tmp.dir, "c.bin", stamp);

    var loaded = (try Prod.load(talloc, tmp.dir, "c.bin", stamp, .{})).?;
    defer loaded.deinit();
    try expectEqual(@as(usize, 0), loaded.slotCount());
    try expectEqual(@as(usize, 0), loaded.len());
}
