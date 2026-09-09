//! Page-aligned, disk-resident vector storage.
//!
//! Every vector starts on a 4096-byte boundary so a single vector read never spans two disk
//! blocks and an update is always a single-block write. Vectors that do not divide the page
//! evenly waste the remainder -- ~24% for the production 768xf32 config -- which is the price
//! of that alignment. The slack is where per-vector metadata and checksums live, with room
//! left over for a future 1-bit quantized code.
//!
//! Nothing is cached in memory: every operation reads or writes the file directly, and a
//! single mutex serializes the whole public surface.
//!
//! v1 is append-only. `rm` leaves a tombstone and the slot is never reused, because reuse
//! requires an id-to-slot index -- without one, a reused slot would either resurrect a dead
//! id or break the arithmetic that makes `get` O(1). The consequence is that a document
//! re-embedded repeatedly grows the file without bound, since `replaceVectors` puts the new
//! vectors before removing the old. Reclaiming that space needs either the index (enabling
//! reuse) or a compaction pass, and both are deliberately deferred.
//!
//! File layout:
//!
//!   +===========+-------------------+-------------------+-------------------+
//!   |  page 0   |      chunk 0      |      chunk 1      |      chunk 2      | ...
//!   |  header   |   (P pages)       |   (P pages)       |   (P pages)       |
//!   +===========+-------------------+-------------------+-------------------+
//!
//! Chunk layout -- vectors from the front, metadata trailer at the back:
//!
//!   +--------+--------+-----+---------+ ......... +------+------+-----+-------+
//!   | vec 0  | vec 1  | ... | vec k-1 |  padding  | m 0  | m 1  | ... | m k-1 |
//!   +--------+--------+-----+---------+ ......... +------+------+-----+-------+
//!   0        S       2S               kS          CHUNK-kM                CHUNK
//!
//! Small vectors pack k-per-chunk with P=1; vectors too large for one page get k=1 and span
//! P pages. Exactly one of k and P is ever greater than 1.

pub const Error = error{
    MultipleRemove,
    OverlappingVectors,
    IncompatibleDatabase,
    UninitializedDocID,
    NoSuchVector,
    Corrupt,
    RangeTooLarge,
};

/// On MacOS the memory page is 16KB, but the disk block is still 4KB, and 4KB is what most
/// other architectures use for both. We align to the lowest common denominator.
pub const PAGE_SZ: usize = 4096;

/// Vector strides are rounded up to this so that every vector in a chunk lands on a
/// SIMD-friendly boundary, not just the first one. Costs nothing at 768 dimensions, where
/// the natural size is already a multiple of 16.
pub const VEC_ALIGN: usize = 16;

const MAGIC: [8]u8 = "DVEVSTOR".*;
const FMT_V: u16 = 1;

const FLAG_USED: u8 = 1 << 0; // slot has been allocated at some point
const FLAG_OCCUPIED: u8 = 1 << 1; // slot currently holds a live vector

/// Immutable description of the on-disk format, written once when the file is created and
/// never rewritten. Keeping every mutable counter out of here means there is no window in
/// which a crash can leave the header disagreeing with the pages behind it: `vec_n` and the
/// next id are derived by scanning metadata trailers at open.
pub const Header = extern struct {
    magic: [8]u8 = MAGIC,
    fmt_v: u16 = FMT_V,
    big_endian: u8 = @intFromBool(native_endian == .big),
    vec_type: u8,
    page_sz: u32,
    vec_sz: u32,
    stride: u32,
    meta_sz: u16,
    vecs_per_chunk: u16,
    pages_per_chunk: u16,
    _pad: u16 = 0,
    crc32: u32 = 0,

    const CRC_COVER = @offsetOf(Header, "crc32");

    fn checksum(self: Header) u32 {
        var h = std.hash.Crc32.init();
        h.update(std.mem.asBytes(&self)[0..CRC_COVER]);
        return h.final();
    }
};

/// Per-vector metadata, one entry per slot, living in the chunk trailer.
///
/// The checksum covers the vector's bytes plus the first 24 bytes of this struct, and
/// deliberately excludes `flags`. That lets `rm` flip a single byte without re-reading and
/// re-hashing the vector, while still catching a torn write that leaves `occupied` set over
/// half-stale vector data. A one-byte flag update cannot itself tear.
pub const VMeta = extern struct {
    doc_id: u64 = 0,
    vec_id: u64 = 0,
    start_i: u32 = 0,
    end_i: u32 = 0,
    crc32: u32 = 0,
    flags: u8 = 0,
    _pad: [3]u8 = .{ 0, 0, 0 },

    const CRC_COVER = @offsetOf(VMeta, "crc32");

    pub fn isUsed(self: VMeta) bool {
        return self.flags & FLAG_USED != 0;
    }
    pub fn isOccupied(self: VMeta) bool {
        return self.flags & FLAG_OCCUPIED != 0;
    }
};

comptime {
    assert(@sizeOf(VMeta) == 32);
    assert(VMeta.CRC_COVER == 24);
}

/// The tag values land in the header on disk, so new types append to the end.
pub const BinaryTypeRepresentation = enum(u8) {
    float32,
    uint8,
    float16,
    int8,

    pub fn to_binary(T: type) BinaryTypeRepresentation {
        return switch (T) {
            f32 => .float32,
            u8 => .uint8,
            f16 => .float16,
            i8 => .int8,
            else => @compileError("no binary representation for " ++ @typeName(T)),
        };
    }
};

/// Comptime resolution of how vectors pack into pages. Solves for vectors and metadata
/// together: picking a vector count first and hoping the remainder fits the metadata fails
/// badly for small vectors, where 4096/12 = 341 vectors leaves 4 bytes for 341 entries.
pub const Layout = struct {
    /// Bytes of actual vector data, before alignment padding.
    vec_bytes: usize,
    /// Distance from one vector to the next, `vec_bytes` rounded up to VEC_ALIGN.
    stride: usize,
    /// Vectors per chunk. 1 when a vector is too large to share a page.
    vecs_per_chunk: usize,
    /// Pages per chunk. 1 unless a single vector needs more than one page.
    pages_per_chunk: usize,
    chunk_bytes: usize,
    /// Offset within a chunk at which the metadata trailer begins.
    meta_off: usize,
    /// Bytes in the chunk claimed by neither vectors nor metadata.
    slack: usize,

    pub fn of(comptime vec_sz: usize, comptime vec_type: type) Layout {
        const vec_bytes = vec_sz * @sizeOf(vec_type);
        const stride = std.mem.alignForward(usize, vec_bytes, VEC_ALIGN);
        const m = @sizeOf(VMeta);

        const packed_k = PAGE_SZ / (stride + m);
        const k = if (packed_k == 0) 1 else packed_k;
        const p = if (packed_k == 0) std.math.divCeil(usize, stride + m, PAGE_SZ) catch unreachable else 1;
        const chunk_bytes = p * PAGE_SZ;
        const meta_off = chunk_bytes - k * m;

        return .{
            .vec_bytes = vec_bytes,
            .stride = stride,
            .vecs_per_chunk = k,
            .pages_per_chunk = p,
            .chunk_bytes = chunk_bytes,
            .meta_off = meta_off,
            .slack = meta_off - k * stride,
        };
    }
};

pub fn VStore(comptime vec_sz: usize, comptime vec_type: type) type {
    const L = Layout.of(vec_sz, vec_type);

    comptime {
        assert(L.chunk_bytes % PAGE_SZ == 0);
        assert(L.stride % VEC_ALIGN == 0);
        assert(L.vecs_per_chunk >= 1);
        // Vectors and their metadata must not overlap.
        assert(L.vecs_per_chunk * L.stride <= L.meta_off);
        // Exactly one of the two regimes applies.
        assert(L.vecs_per_chunk == 1 or L.pages_per_chunk == 1);
        // The trailer is 8-aligned within the chunk, so VMeta can be read in place.
        assert(L.meta_off % @alignOf(VMeta) == 0);
    }

    return struct {
        const Self = @This();

        pub const LAYOUT = L;
        pub const VECS_PER_CHUNK = L.vecs_per_chunk;
        pub const PAGES_PER_CHUNK = L.pages_per_chunk;
        pub const CHUNK_BYTES = L.chunk_bytes;
        pub const STRIDE = L.stride;
        pub const VEC_BYTES = L.vec_bytes;

        /// The element type varies with the build's quantization setting, so callers that
        /// hold vectors of their own should name these rather than assume f32.
        pub const Vector = @Vector(vec_sz, vec_type);
        /// The array form is what crosses the API boundary. A @Vector(768, f32) is padded to
        /// 4096 bytes and aligned to 4096, so passing one by value costs more than the
        /// vector itself; the array is the bare 3072 bytes and coerces to Vector on use.
        pub const Array = [vec_sz]vec_type;

        pub const Row = struct {
            vec_id: VectorID = 0,
            doc_id: DocID = 0,
            start_i: usize = 0,
            end_i: usize = 0,
        };

        pub const PutMeta = struct {
            doc_id: DocID,
            start_i: usize,
            end_i: usize,
        };

        pub const SearchEntry = struct {
            row: Row = .{},
            similarity: f32 = 0.0,
        };

        pub const default_filename = std.fmt.comptimePrint(
            "{d}_{s}.db",
            .{ vec_sz, @typeName(vec_type) },
        );

        pub const Opts = struct {
            path: []const u8 = default_filename,
        };

        allocator: std.mem.Allocator,
        dir: std.fs.Dir,
        file: pfile.File,
        /// Held across every public method. Callers store a VStore by value, so it is copied
        /// once out of `init` -- before any thread can contend for it -- and must not be
        /// copied again afterwards.
        mutex: std.Thread.Mutex = .{},
        /// Live vectors. Derived at open, maintained thereafter.
        vec_n: usize = 0,
        /// Slots ever allocated. v1 is append-only, so this only grows; freed slots become
        /// tombstones rather than being reused, because reuse would need the id-to-slot
        /// index we have deliberately deferred.
        slot_n: usize = 0,

        // ******************************************************************** Slot addressing
        /// v1 hands out ids equal to the slot index, but that is an allocator detail and not
        /// a promise: `vec_id` is written into every metadata entry, so once an index exists
        /// slots can be reused and relocated without a format change.
        fn slotOf(id: VectorID) usize {
            return id;
        }
        fn idOf(slot: usize) VectorID {
            return slot;
        }

        fn chunkOff(chunk_i: usize) u64 {
            return PAGE_SZ + chunk_i * L.chunk_bytes;
        }
        fn vecOff(slot: usize) u64 {
            return chunkOff(slot / L.vecs_per_chunk) + (slot % L.vecs_per_chunk) * L.stride;
        }
        fn metaOff(slot: usize) u64 {
            return chunkOff(slot / L.vecs_per_chunk) +
                L.meta_off + (slot % L.vecs_per_chunk) * @sizeOf(VMeta);
        }
        fn chunkCount(slot_count: usize) usize {
            return std.math.divCeil(usize, slot_count, L.vecs_per_chunk) catch unreachable;
        }

        fn crcOf(vec: *const Array, m: VMeta) u32 {
            var h = std.hash.Crc32.init();
            h.update(std.mem.asBytes(vec)[0..L.vec_bytes]);
            h.update(std.mem.asBytes(&m)[0..VMeta.CRC_COVER]);
            return h.final();
        }

        // ************************************************************************* Lifecycle
        pub fn init(allocator: std.mem.Allocator, dir: std.fs.Dir, opts: Opts) !Self {
            const file = try pfile.File.openAt(@intCast(dir.fd), opts.path, .{});
            errdefer file.close();

            var self = Self{ .allocator = allocator, .dir = dir, .file = file };

            const size = try file.size();
            if (size == 0) {
                try self.writeHeader();
            } else {
                try self.checkHeader();
                try self.scan();
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.file.close();
        }

        fn writeHeader(self: *Self) !void {
            var page = [_]u8{0} ** PAGE_SZ;
            var h = Header{
                .vec_type = @intFromEnum(BinaryTypeRepresentation.to_binary(vec_type)),
                .page_sz = PAGE_SZ,
                .vec_sz = vec_sz,
                .stride = @intCast(L.stride),
                .meta_sz = @sizeOf(VMeta),
                .vecs_per_chunk = @intCast(L.vecs_per_chunk),
                .pages_per_chunk = @intCast(L.pages_per_chunk),
            };
            h.crc32 = h.checksum();
            @memcpy(page[0..@sizeOf(Header)], std.mem.asBytes(&h));
            try self.file.writeAt(&page, 0);
        }

        fn checkHeader(self: *Self) !void {
            var buf: [@sizeOf(Header)]u8 = undefined;
            const n = self.file.readAt(&buf, 0) catch return Error.IncompatibleDatabase;
            if (n != buf.len) return Error.IncompatibleDatabase;
            const h: *const Header = @ptrCast(@alignCast(&buf));

            // Cross-endian files are rejected rather than byte-swapped: the format is
            // machine-local in practice, and refusing is honest about what we support.
            const ok = std.mem.eql(u8, &h.magic, &MAGIC) and
                h.fmt_v == FMT_V and
                h.big_endian == @intFromBool(native_endian == .big) and
                h.vec_type == @intFromEnum(BinaryTypeRepresentation.to_binary(vec_type)) and
                h.page_sz == PAGE_SZ and
                h.vec_sz == vec_sz and
                h.stride == L.stride and
                h.meta_sz == @sizeOf(VMeta) and
                h.vecs_per_chunk == L.vecs_per_chunk and
                h.pages_per_chunk == L.pages_per_chunk and
                h.crc32 == h.checksum();

            if (!ok) {
                std.log.warn(
                    "Incompatible database: fmt_v={d} (want {d}), vec_sz={d} (want {d}), stride={d} (want {d})",
                    .{ h.fmt_v, FMT_V, h.vec_sz, vec_sz, h.stride, L.stride },
                );
                return Error.IncompatibleDatabase;
            }
        }

        /// Rebuilds the live count and the slot high-water mark by walking every chunk's
        /// metadata trailer. This is what buys us an immutable header. It costs one read per
        /// chunk at open, which a sidecar index would later make a cold-start-only cost.
        fn scan(self: *Self) !void {
            const size = try self.file.size();
            // Nothing to scan unless the file reaches at least the first chunk's trailer.
            if (size <= chunkOff(0) + L.meta_off) return;
            const n_chunks = (size - PAGE_SZ) / L.chunk_bytes;

            var trailer: [L.vecs_per_chunk * @sizeOf(VMeta)]u8 align(@alignOf(VMeta)) = undefined;
            var vec_n: usize = 0;
            var slot_n: usize = 0;

            for (0..n_chunks) |c| {
                const got = try self.file.readAt(&trailer, chunkOff(c) + L.meta_off);
                if (got != trailer.len) break; // truncated tail: stop where the file stops
                for (0..L.vecs_per_chunk) |i| {
                    const m = metaAt(&trailer, i);
                    if (!m.isUsed()) continue;
                    const slot = c * L.vecs_per_chunk + i;
                    slot_n = slot + 1;
                    if (m.isOccupied()) vec_n += 1;
                }
            }

            self.vec_n = vec_n;
            self.slot_n = slot_n;
        }

        fn metaAt(trailer: []align(@alignOf(VMeta)) const u8, i: usize) *const VMeta {
            return @ptrCast(@alignCast(trailer[i * @sizeOf(VMeta) ..].ptr));
        }

        /// Barrier, not a save: there is no in-memory copy to write back, so this only asks
        /// the OS to make already-written bytes durable. Real transactions are a later want.
        pub fn flush(self: *Self) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.file.sync();
        }

        pub fn len(self: *Self) usize {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.vec_n;
        }

        // ************************************************************************* Core ops
        pub fn put(self: *Self, meta: PutMeta, vec: *const Array) !VectorID {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.putLocked(meta, vec);
        }

        fn putLocked(self: *Self, meta: PutMeta, vec: *const Array) !VectorID {
            // Ranges are stored as u32 to keep VMeta at 32 bytes. `vector.zig` already caps
            // documents at maxInt(u32), so this is an assertion of an existing invariant --
            // but it is the caller's data, so it returns an error rather than trapping.
            if (meta.start_i > std.math.maxInt(u32) or meta.end_i > std.math.maxInt(u32)) {
                return Error.RangeTooLarge;
            }
            const slot = self.slot_n;
            const id = idOf(slot);

            // Keep the file a whole number of chunks so the trailer of the last chunk always
            // exists to be read back.
            const want = chunkOff(chunkCount(slot + 1));
            if (try self.file.size() < want) try self.file.setSize(want);

            var m = VMeta{
                .doc_id = meta.doc_id,
                .vec_id = id,
                .start_i = @intCast(meta.start_i),
                .end_i = @intCast(meta.end_i),
                .flags = FLAG_USED | FLAG_OCCUPIED,
            };
            m.crc32 = crcOf(vec, m);

            // Vector first, metadata second: a crash between them leaves a slot that is not
            // marked used, which the open scan simply skips.
            try self.file.writeAt(std.mem.asBytes(vec)[0..L.vec_bytes], vecOff(slot));
            try self.file.writeAt(std.mem.asBytes(&m), metaOff(slot));

            self.slot_n = slot + 1;
            self.vec_n += 1;
            return id;
        }

        fn readMeta(self: *Self, slot: usize) !VMeta {
            var m: VMeta = undefined;
            const n = try self.file.readAt(std.mem.asBytes(&m), metaOff(slot));
            if (n != @sizeOf(VMeta)) return Error.Corrupt;
            return m;
        }

        /// Null for an id that was never handed out or whose vector has been removed.
        pub fn get(self: *Self, id: VectorID) !?Row {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.getLocked(id);
        }

        fn getLocked(self: *Self, id: VectorID) !?Row {
            const slot = slotOf(id);
            if (slot >= self.slot_n) return null;
            const m = try self.readMeta(slot);
            if (!m.isOccupied()) return null;
            return rowOf(m);
        }

        fn rowOf(m: VMeta) Row {
            return .{
                .vec_id = @intCast(m.vec_id),
                .doc_id = m.doc_id,
                .start_i = m.start_i,
                .end_i = m.end_i,
            };
        }

        /// Reads into the caller's buffer and verifies the checksum. `search` deliberately
        /// skips verification on its inner loop; `validate` checks everything.
        pub fn getVec(self: *Self, id: VectorID, out: *Array) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.getVecLocked(id, out);
        }

        fn getVecLocked(self: *Self, id: VectorID, out: *Array) !void {
            const slot = slotOf(id);
            if (slot >= self.slot_n) return Error.NoSuchVector;
            const m = try self.readMeta(slot);
            if (!m.isOccupied()) return Error.NoSuchVector;

            const n = try self.file.readAt(std.mem.asBytes(out)[0..L.vec_bytes], vecOff(slot));
            if (n != L.vec_bytes) return Error.Corrupt;
            if (crcOf(out, m) != m.crc32) return Error.Corrupt;
        }

        pub fn rm(self: *Self, id: VectorID) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.rmLocked(id);
        }

        fn rmLocked(self: *Self, id: VectorID) !void {
            const slot = slotOf(id);
            if (slot >= self.slot_n) return Error.MultipleRemove;
            var m = try self.readMeta(slot);
            if (!m.isOccupied()) return Error.MultipleRemove;
            assert(self.vec_n > 0);

            // Only the flags byte changes, and it is outside the checksum's coverage, so the
            // vector never has to be re-read. The bytes stay on disk as a tombstone.
            m.flags &= ~FLAG_OCCUPIED;
            try self.file.writeAt(
                std.mem.asBytes(&m)[@offsetOf(VMeta, "flags")..][0..1],
                metaOff(slot) + @offsetOf(VMeta, "flags"),
            );
            self.vec_n -= 1;
        }

        /// Generates a new ID for an existing VectorRow.
        pub fn copy(self: *Self, id: VectorID) !VectorID {
            self.mutex.lock();
            defer self.mutex.unlock();

            const row = (try self.getLocked(id)) orelse return Error.NoSuchVector;
            var vec: Array = undefined;
            try self.getVecLocked(id, &vec);
            return self.putLocked(.{
                .doc_id = row.doc_id,
                .start_i = row.start_i,
                .end_i = row.end_i,
            }, &vec);
        }

        // *********************************************************************** Whole-store
        /// One chunk-sized aligned buffer, so a scan is one read per chunk rather than one
        /// per vector and per trailer.
        fn allocChunkBuf(self: *Self) ![]align(VEC_ALIGN) u8 {
            return self.allocator.alignedAlloc(u8, .@"16", L.chunk_bytes);
        }

        fn vecAt(buf: []align(VEC_ALIGN) const u8, i: usize) *const Array {
            return @ptrCast(@alignCast(buf[i * L.stride ..][0..L.vec_bytes]));
        }

        pub fn search(self: *Self, query: *const Array, buf: []SearchEntry, threshold: f32) !usize {
            const zone = tracy.beginZone(@src(), .{ .name = "vstore.zig:search" });
            defer zone.end();

            self.mutex.lock();
            defer self.mutex.unlock();

            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            const Cand = struct {
                row: Row,
                // f32 regardless of vec_type: storedDot reports similarity in the units of
                // the original embeddings, which is also what SearchEntry carries.
                sim: f32,

                fn order(_: void, a: @This(), b: @This()) std.math.Order {
                    return std.math.order(b.sim, a.sim);
                }
            };
            var pq = std.PriorityQueue(Cand, void, Cand.order).init(arena.allocator(), undefined);

            const chunk = try self.allocChunkBuf();
            defer self.allocator.free(chunk);

            const n_chunks = chunkCount(self.slot_n);
            for (0..n_chunks) |c| {
                const got = try self.file.readAt(chunk, chunkOff(c));
                if (got != chunk.len) break;
                const trailer = chunk[L.meta_off..];
                for (0..L.vecs_per_chunk) |i| {
                    const slot = c * L.vecs_per_chunk + i;
                    if (slot >= self.slot_n) break;
                    const m = metaAt(@alignCast(trailer), i);
                    if (!m.isOccupied()) continue;
                    const sim = storedDot(vec_sz, vec_type, vecAt(chunk, i).*, query.*);
                    // The row is already in hand from the trailer we just read; re-reading it
                    // per result would cost a syscall each.
                    if (sim > threshold) try pq.add(.{ .row = rowOf(m.*), .sim = sim });
                }
            }

            var n: usize = 0;
            while (pq.removeOrNull()) |cand| : (n += 1) {
                if (n >= buf.len) {
                    std.log.debug("Results capped", .{});
                    break;
                }
                buf[n] = .{ .row = cand.row, .similarity = cand.sim };
            }
            return n;
        }

        /// Returns all vectors for a given doc_id, sorted by start_i.
        pub fn vecsForDoc(self: *Self, allocator: std.mem.Allocator, doc_id: DocID) ![]Row {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.vecsForDocLocked(allocator, doc_id);
        }

        fn vecsForDocLocked(self: *Self, allocator: std.mem.Allocator, doc_id: DocID) ![]Row {
            var results: std.ArrayList(Row) = .{};
            errdefer results.deinit(allocator);

            var trailer: [L.vecs_per_chunk * @sizeOf(VMeta)]u8 align(@alignOf(VMeta)) = undefined;
            const n_chunks = chunkCount(self.slot_n);
            for (0..n_chunks) |c| {
                const got = try self.file.readAt(&trailer, chunkOff(c) + L.meta_off);
                if (got != trailer.len) break;
                for (0..L.vecs_per_chunk) |i| {
                    const slot = c * L.vecs_per_chunk + i;
                    if (slot >= self.slot_n) break;
                    const m = metaAt(&trailer, i);
                    if (!m.isOccupied() or m.doc_id != doc_id) continue;
                    try results.append(allocator, rowOf(m.*));
                }
            }

            const items = try results.toOwnedSlice(allocator);
            std.sort.insertion(Row, items, {}, struct {
                fn lessThan(_: void, a: Row, b: Row) bool {
                    return a.start_i < b.start_i;
                }
            }.lessThan);
            return items;
        }

        /// Removes all vectors for a given doc_id.
        pub fn rmByDocId(self: *Self, doc_id: DocID) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const rows = try self.vecsForDocLocked(self.allocator, doc_id);
            defer self.allocator.free(rows);
            for (rows) |row| try self.rmLocked(row.vec_id);
        }

        /// Validates the store is in a good state by:
        /// - verifying every live vector's checksum against its bytes on disk
        /// - checking every live vector is L2 normalized
        /// - checking every DocID is initialized
        /// - per-doc checking that no two ranges collide
        pub fn validate(self: *Self) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();
            const a = arena.allocator();

            const chunk = try self.allocChunkBuf();
            defer self.allocator.free(chunk);

            var docs_seen = std.AutoHashMap(DocID, void).init(a);
            const n_chunks = chunkCount(self.slot_n);
            for (0..n_chunks) |c| {
                const got = try self.file.readAt(chunk, chunkOff(c));
                if (got != chunk.len) return Error.Corrupt;
                const trailer = chunk[L.meta_off..];
                for (0..L.vecs_per_chunk) |i| {
                    const slot = c * L.vecs_per_chunk + i;
                    if (slot >= self.slot_n) break;
                    const m = metaAt(@alignCast(trailer), i);
                    if (!m.isOccupied()) continue;

                    if (m.doc_id == 0) return Error.UninitializedDocID;
                    const vec = vecAt(chunk, i);
                    if (crcOf(vec, m.*) != m.crc32) {
                        std.log.warn("vstore: checksum mismatch at slot {d}", .{slot});
                        return Error.Corrupt;
                    }
                    try validateL2(vec_sz, vec_type, vec.*);
                    try docs_seen.put(m.doc_id, {});
                }
            }

            var it = docs_seen.keyIterator();
            while (it.next()) |doc_id| {
                const rows = try self.vecsForDocLocked(a, doc_id.*);
                for (0..rows.len) |i| {
                    for (i + 1..rows.len) |j| {
                        const x = rows[i];
                        const y = rows[j];
                        if (x.start_i < y.end_i and y.start_i < x.end_i) {
                            return Error.OverlappingVectors;
                        }
                    }
                }
            }
        }
    };
}

// **************************************************************************************** Vectors
pub fn dot(comptime N: u32, comptime T: type, a: @Vector(N, T), b: @Vector(N, T)) T {
    return @reduce(.Add, a * b);
}

pub fn magnitude(comptime N: u32, comptime T: type, a: @Vector(N, T)) T {
    return @sqrt(@reduce(.Add, a * a));
}

fn is_zero(comptime N: u32, comptime T: type, a: @Vector(N, T)) bool {
    const zero_vec: @Vector(N, T) = @splat(0);
    return @reduce(.And, a == zero_vec);
}

pub fn cosine_similarity(comptime N: u32, comptime T: type, a: @Vector(N, T), b: @Vector(N, T)) T {
    if (is_zero(N, T, a) or is_zero(N, T, b)) return 0;
    return dot(N, T, a, b) / (magnitude(N, T, a) * magnitude(N, T, b));
}

// ****************************************************************************************** Tests
const TestT = f32;
const TestN = 3;
const TestVecType = @Vector(TestN, TestT);
const TestArray = [TestN]TestT;
const TestStorage = VStore(TestN, TestT);

const DB = "test.db";

fn open(dir: std.fs.Dir) !TestStorage {
    return TestStorage.init(testing_allocator, dir, .{ .path = DB });
}

fn expectVecError(inst: *TestStorage, id: VectorID, want: anyerror) !void {
    var scratch: TestArray = undefined;
    try std.testing.expectError(want, inst.getVec(id, &scratch));
}

fn expectVec(inst: *TestStorage, id: VectorID, want: TestArray) !void {
    var got: TestArray = undefined;
    try inst.getVec(id, &got);
    try expect(@reduce(.And, @as(TestVecType, want) == @as(TestVecType, got)));
}

// ************************************************************************************ Layout math
test "layout: packing math across configs" {
    // 12 bytes of data padded to a 16-byte stride: 85 vectors and their 32-byte metadata
    // entries fill 4080 of the page. The pre-alignment count of 93 would have left 4 bytes
    // for 93 metadata entries, which is the bug this formula exists to avoid.
    const tiny = Layout.of(3, f32);
    try expectEqual(@as(usize, 12), tiny.vec_bytes);
    try expectEqual(@as(usize, 16), tiny.stride);
    try expectEqual(@as(usize, 85), tiny.vecs_per_chunk);
    try expectEqual(@as(usize, 1), tiny.pages_per_chunk);
    try expectEqual(@as(usize, 16), tiny.slack);

    // Production config: one vector per page, ~24% of the page unused. That slack is the
    // price of page alignment, and where a future 1-bit code (96 bytes) would live.
    const prod = Layout.of(768, f32);
    try expectEqual(@as(usize, 3072), prod.stride);
    try expectEqual(@as(usize, 1), prod.vecs_per_chunk);
    try expectEqual(@as(usize, 1), prod.pages_per_chunk);
    try expectEqual(@as(usize, 992), prod.slack);

    const f16_cfg = Layout.of(768, f16);
    try expectEqual(@as(usize, 2), f16_cfg.vecs_per_chunk);
    try expectEqual(@as(usize, 960), f16_cfg.slack);

    const i8_cfg = Layout.of(768, i8);
    try expectEqual(@as(usize, 5), i8_cfg.vecs_per_chunk);
    try expectEqual(@as(usize, 96), i8_cfg.slack);

    // 4064 bytes of vector plus 32 of metadata is exactly one page, with nothing wasted.
    const exact = Layout.of(1016, f32);
    try expectEqual(@as(usize, 1), exact.vecs_per_chunk);
    try expectEqual(@as(usize, 1), exact.pages_per_chunk);
    try expectEqual(@as(usize, 0), exact.slack);

    // Too large for one page: k collapses to 1 and the chunk spans P pages instead.
    const multi = Layout.of(1536, f32);
    try expectEqual(@as(usize, 6144), multi.stride);
    try expectEqual(@as(usize, 1), multi.vecs_per_chunk);
    try expectEqual(@as(usize, 2), multi.pages_per_chunk);
    try expectEqual(@as(usize, 8192), multi.chunk_bytes);
}

test "layout: invariants hold for every dimension up to 2048" {
    @setEvalBranchQuota(200_000);
    inline for (.{ f32, f16, i8 }) |T| {
        comptime var n: usize = 1;
        inline while (n <= 2048) : (n += 1) {
            const l = comptime Layout.of(n, T);
            comptime {
                assert(l.chunk_bytes % PAGE_SZ == 0);
                assert(l.stride % VEC_ALIGN == 0);
                assert(l.stride >= l.vec_bytes);
                assert(l.vecs_per_chunk >= 1);
                assert(l.vecs_per_chunk * l.stride + l.vecs_per_chunk * @sizeOf(VMeta) <= l.chunk_bytes);
                assert(l.vecs_per_chunk == 1 or l.pages_per_chunk == 1);
                assert(l.meta_off % @alignOf(VMeta) == 0);
            }
        }
    }
}

// ************************************************************************************* Basic ops
test "put / get" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    try expectEqual(@as(usize, 0), inst.len());
    const id = try inst.put(.{ .doc_id = 42, .start_i = 0, .end_i = 10 }, &.{ 1, 0, 0 });
    try expectEqual(@as(usize, 1), inst.len());

    const row = (try inst.get(id)).?;
    try expectEqual(@as(DocID, 42), row.doc_id);
    try expectEqual(@as(usize, 0), row.start_i);
    try expectEqual(@as(usize, 10), row.end_i);
    try expectEqual(id, row.vec_id);
    try expectVec(&inst, id, .{ 1, 0, 0 });
}

test "get of an id that was never handed out is null" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    try expect((try inst.get(0)) == null);
    try expect((try inst.get(99)) == null);
    try expectVecError(&inst, 0, Error.NoSuchVector);
}

test "rm makes the row disappear, and removing twice is an error" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    const id = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 5 }, &.{ 1, 0, 0 });
    try inst.rm(id);
    try expectEqual(@as(usize, 0), inst.len());
    try expect((try inst.get(id)) == null);
    try std.testing.expectError(Error.MultipleRemove, inst.rm(id));
}

test "ids stay distinct across a delete" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    const a = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 5 }, &.{ 1, 0, 0 });
    try inst.rm(a);
    const b = try inst.put(.{ .doc_id = 2, .start_i = 0, .end_i = 5 }, &.{ 0, 1, 0 });

    // v1 is append-only, so a freed slot is never reused and a stale id can never resolve to
    // a different vector.
    try expect(a != b);
    try expect((try inst.get(a)) == null);
    try expectEqual(@as(DocID, 2), (try inst.get(b)).?.doc_id);
}

test "copy" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    const old_id = try inst.put(.{ .doc_id = 42, .start_i = 5, .end_i = 15 }, &.{ 1, 0, 0 });
    const new_id = try inst.copy(old_id);
    try expect(old_id != new_id);

    const a = (try inst.get(old_id)).?;
    const b = (try inst.get(new_id)).?;
    try expectEqual(a.doc_id, b.doc_id);
    try expectEqual(a.start_i, b.start_i);
    try expectEqual(a.end_i, b.end_i);
    try expectVec(&inst, new_id, .{ 1, 0, 0 });
}

// *************************************************************************************** Reopen
test "reopen: a single vector survives" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var id: VectorID = undefined;
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        id = try inst.put(.{ .doc_id = 100, .start_i = 5, .end_i = 15 }, &.{ 1, 0, 0 });
        try inst.flush();
    }

    var inst2 = try open(tmpD.dir);
    defer inst2.deinit();
    try expectEqual(@as(usize, 1), inst2.len());
    const row = (try inst2.get(id)).?;
    try expectEqual(@as(DocID, 100), row.doc_id);
    try expectEqual(@as(usize, 5), row.start_i);
    try expectEqual(@as(usize, 15), row.end_i);
    try expectVec(&inst2, id, .{ 1, 0, 0 });
}

const TestRow = struct { vec: TestArray, doc_id: DocID, start_i: usize, end_i: usize };
const four_rows: [4]TestRow = .{
    .{ .vec = .{ 1, 0, 0 }, .doc_id = 1, .start_i = 0, .end_i = 10 },
    .{ .vec = .{ 0, 1, 0 }, .doc_id = 2, .start_i = 10, .end_i = 20 },
    .{ .vec = .{ 0, 0, 1 }, .doc_id = 3, .start_i = 20, .end_i = 30 },
    .{ .vec = .{ 0.6, 0.8, 0 }, .doc_id = 4, .start_i = 30, .end_i = 40 },
};

test "reopen: multiple vectors survive" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var ids: [4]VectorID = undefined;
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        for (four_rows, 0..) |r, i| {
            ids[i] = try inst.put(.{ .doc_id = r.doc_id, .start_i = r.start_i, .end_i = r.end_i }, &r.vec);
        }
        try expectEqual(@as(usize, 4), inst.len());
        try inst.flush();
    }

    var inst2 = try open(tmpD.dir);
    defer inst2.deinit();
    try expectEqual(@as(usize, 4), inst2.len());
    for (four_rows, ids) |r, id| {
        const row = (try inst2.get(id)).?;
        try expectEqual(r.doc_id, row.doc_id);
        try expectEqual(r.start_i, row.start_i);
        try expectEqual(r.end_i, row.end_i);
        try expectVec(&inst2, id, r.vec);
    }
    try inst2.validate();
}

test "reopen: tombstones survive and do not inflate the count" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var ids: [4]VectorID = undefined;
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        for (four_rows, 0..) |r, i| {
            ids[i] = try inst.put(.{ .doc_id = r.doc_id, .start_i = r.start_i, .end_i = r.end_i }, &r.vec);
        }
        try inst.rm(ids[0]);
        try inst.rm(ids[2]);
        try expectEqual(@as(usize, 2), inst.len());
        try inst.flush();
    }

    var inst2 = try open(tmpD.dir);
    defer inst2.deinit();
    try expectEqual(@as(usize, 2), inst2.len());
    try expect((try inst2.get(ids[0])) == null);
    try expect((try inst2.get(ids[1])) != null);
    try expect((try inst2.get(ids[2])) == null);
    try expect((try inst2.get(ids[3])) != null);

    // A removed slot must not come back to life just because the bytes are still on disk.
    try expectVecError(&inst2, ids[0], Error.NoSuchVector);

    // The high-water mark is recovered too, so new ids do not collide with tombstoned ones.
    const fresh = try inst2.put(.{ .doc_id = 9, .start_i = 40, .end_i = 50 }, &.{ 1, 0, 0 });
    for (ids) |id| try expect(fresh != id);
}

test "opening a non-existent db creates an empty one" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var inst = try TestStorage.init(testing_allocator, tmpD.dir, .{ .path = "does-not-exist.db" });
    defer inst.deinit();
    try expectEqual(@as(usize, 0), inst.len());
    try inst.validate();
}

test "a db written for a different vector type is rejected" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        _ = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 1 }, &.{ 1, 0, 0 });
    }

    const Other = VStore(4, f32);
    try std.testing.expectError(
        Error.IncompatibleDatabase,
        Other.init(testing_allocator, tmpD.dir, .{ .path = DB }),
    );
}

// ************************************************************************** On-disk format checks
test "format: the file is a whole number of pages and vectors are page aligned" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    for (0..200) |i| {
        _ = try inst.put(.{ .doc_id = @intCast(i + 1), .start_i = i, .end_i = i + 1 }, &.{ 1, 0, 0 });
    }

    const size = try inst.file.size();
    try expectEqual(@as(u64, 0), size % PAGE_SZ);
    // Header page plus enough whole chunks for 200 slots at 85 per chunk.
    try expectEqual(PAGE_SZ + 3 * TestStorage.CHUNK_BYTES, size);

    // The first vector of every chunk starts exactly on a page boundary.
    try expectEqual(@as(u64, 0), TestStorage.vecOff(0) % PAGE_SZ);
    try expectEqual(@as(u64, 0), TestStorage.vecOff(TestStorage.VECS_PER_CHUNK) % PAGE_SZ);
    // And no vector ever straddles the end of its chunk.
    const last = TestStorage.VECS_PER_CHUNK - 1;
    try expect(TestStorage.vecOff(last) % TestStorage.CHUNK_BYTES + TestStorage.VEC_BYTES <=
        TestStorage.LAYOUT.meta_off);
}

test "format: a flipped byte in the vector region is caught by the checksum" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var id: VectorID = undefined;
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        id = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 10 }, &.{ 1, 0, 0 });
        try inst.validate();
        try inst.flush();
    }

    {
        const f = try pfile.File.openAt(@intCast(tmpD.dir.fd), DB, .{ .create = false });
        defer f.close();
        var byte: [1]u8 = undefined;
        _ = try f.readAt(&byte, PAGE_SZ);
        byte[0] ^= 0xFF;
        try f.writeAt(&byte, PAGE_SZ);
    }

    var inst2 = try open(tmpD.dir);
    defer inst2.deinit();
    try std.testing.expectError(Error.Corrupt, inst2.validate());
    try expectVecError(&inst2, id, Error.Corrupt);
}

test "format: a truncated tail does not take the whole store down" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        for (0..100) |i| {
            _ = try inst.put(.{ .doc_id = @intCast(i + 1), .start_i = i, .end_i = i + 1 }, &.{ 1, 0, 0 });
        }
        try inst.flush();
    }

    // Lop off the second chunk entirely, as an interrupted extend would.
    {
        const f = try pfile.File.openAt(@intCast(tmpD.dir.fd), DB, .{ .create = false });
        defer f.close();
        try f.setSize(PAGE_SZ + TestStorage.CHUNK_BYTES);
    }

    var inst2 = try open(tmpD.dir);
    defer inst2.deinit();
    try expectEqual(@as(usize, TestStorage.VECS_PER_CHUNK), inst2.len());
    try inst2.validate();
}

// ***************************************************************************** Alternate layouts
test "multi-page vectors: one vector spanning two pages" {
    const N = 1536;
    const Multi = VStore(N, f32);
    try expectEqual(@as(usize, 2), Multi.PAGES_PER_CHUNK);
    try expectEqual(@as(usize, 1), Multi.VECS_PER_CHUNK);

    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try Multi.init(testing_allocator, tmpD.dir, .{ .path = DB });
    defer inst.deinit();

    var vec: [N]f32 = @splat(1.0 / @sqrt(@as(f32, N)));
    const id = try inst.put(.{ .doc_id = 7, .start_i = 0, .end_i = 3 }, &vec);
    vec[0] = 0;

    var got: [N]f32 = undefined;
    try inst.getVec(id, &got);
    try expect(got[0] != 0);
    try expectEqual(@as(usize, 2), got.len / (N / 2));
    try inst.validate();

    try expectEqual(PAGE_SZ + Multi.CHUNK_BYTES, try inst.file.size());
}

test "zero-slack layout: vector plus metadata exactly fills a page" {
    const N = 1016;
    const Exact = VStore(N, f32);
    try expectEqual(@as(usize, 0), Exact.LAYOUT.slack);

    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try Exact.init(testing_allocator, tmpD.dir, .{ .path = DB });
    defer inst.deinit();

    const vec: [N]f32 = @splat(1.0 / @sqrt(@as(f32, N)));
    const id = try inst.put(.{ .doc_id = 3, .start_i = 0, .end_i = 1 }, &vec);
    var got: [N]f32 = undefined;
    try inst.getVec(id, &got);
    try expectEqual(vec[0], got[0]);
    try expectEqual(vec[N - 1], got[N - 1]);
    try inst.validate();
}

test "many vectors per chunk: the chunk boundary is not a special case" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    // Straddle the first chunk boundary in both directions.
    const n = TestStorage.VECS_PER_CHUNK + 3;
    var ids: [TestStorage.VECS_PER_CHUNK + 3]VectorID = undefined;
    for (0..n) |i| {
        const v: TestArray = .{ @floatFromInt(i % 2), @floatFromInt((i + 1) % 2), 0 };
        ids[i] = try inst.put(.{ .doc_id = @intCast(i + 1), .start_i = i, .end_i = i + 1 }, &v);
    }
    for (0..n) |i| {
        const row = (try inst.get(ids[i])).?;
        try expectEqual(@as(DocID, @intCast(i + 1)), row.doc_id);
        try expectVec(&inst, ids[i], .{ @floatFromInt(i % 2), @floatFromInt((i + 1) % 2), 0 });
    }
    try inst.validate();
}

// **************************************************************************************** Search
test "search returns rows ordered by similarity" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    _ = try inst.put(.{ .doc_id = 10, .start_i = 0, .end_i = 5 }, &.{ 1, 0, 0 });
    _ = try inst.put(.{ .doc_id = 20, .start_i = 5, .end_i = 10 }, &.{ 0.9, 0.1, 0 });
    _ = try inst.put(.{ .doc_id = 30, .start_i = 10, .end_i = 15 }, &.{ 0, 1, 0 });

    var results: [10]TestStorage.SearchEntry = undefined;
    const n = try inst.search(&.{ 1, 0, 0 }, &results, 0.5);

    try expectEqual(@as(usize, 2), n);
    try expectEqual(@as(DocID, 10), results[0].row.doc_id);
    try expectEqual(@as(DocID, 20), results[1].row.doc_id);
    try expect(results[0].similarity > results[1].similarity);
}

test "search skips removed vectors" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    const a = try inst.put(.{ .doc_id = 10, .start_i = 0, .end_i = 5 }, &.{ 1, 0, 0 });
    _ = try inst.put(.{ .doc_id = 20, .start_i = 5, .end_i = 10 }, &.{ 0.9, 0.1, 0 });

    var results: [10]TestStorage.SearchEntry = undefined;
    try expectEqual(@as(usize, 2), try inst.search(&.{ 1, 0, 0 }, &results, 0.5));
    try inst.rm(a);
    try expectEqual(@as(usize, 1), try inst.search(&.{ 1, 0, 0 }, &results, 0.5));
    try expectEqual(@as(DocID, 20), results[0].row.doc_id);
}

test "search hugebuf" {
    const HugeN = 1000;
    const HugeStorage = VStore(HugeN, f32);

    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try HugeStorage.init(testing_allocator, tmpD.dir, .{ .path = DB });
    defer inst.deinit();

    for (0..1000) |i| {
        var vec: [HugeN]f32 = @splat(0);
        vec[i % HugeN] = 1.0; // Each vector has a single 1.0 at position i%N
        _ = try inst.put(.{ .doc_id = @intCast(i + 1), .start_i = i * 10, .end_i = (i + 1) * 10 }, &vec);
    }

    var results: [1000]HugeStorage.SearchEntry = undefined;
    var query: [HugeN]f32 = @splat(0);
    query[0] = 1.0;

    const n = try inst.search(&query, &results, 0.0);
    try expectEqual(@as(usize, 1), n);
    try expectEqual(@as(DocID, 1), results[0].row.doc_id);
    try expectEqual(@as(f32, 1.0), results[0].similarity);
}

// ****************************************************************************** Doc-level queries
test "vecsForDoc returns only that doc's rows, sorted by start_i" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    _ = try inst.put(.{ .doc_id = 1, .start_i = 20, .end_i = 30 }, &.{ 1, 0, 0 });
    _ = try inst.put(.{ .doc_id = 2, .start_i = 0, .end_i = 5 }, &.{ 0, 1, 0 });
    _ = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 10 }, &.{ 0, 0, 1 });

    const rows = try inst.vecsForDoc(testing_allocator, 1);
    defer testing_allocator.free(rows);
    try expectEqual(@as(usize, 2), rows.len);
    try expectEqual(@as(usize, 0), rows[0].start_i);
    try expectEqual(@as(usize, 20), rows[1].start_i);
}

test "rmByDocId" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    _ = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 5 }, &.{ 1, 0, 0 });
    _ = try inst.put(.{ .doc_id = 1, .start_i = 5, .end_i = 10 }, &.{ 0, 1, 0 });
    _ = try inst.put(.{ .doc_id = 2, .start_i = 0, .end_i = 5 }, &.{ 0, 0, 1 });
    try expectEqual(@as(usize, 3), inst.len());

    try inst.rmByDocId(1);
    try expectEqual(@as(usize, 1), inst.len());

    const rows = try inst.vecsForDoc(testing_allocator, 2);
    defer testing_allocator.free(rows);
    try expectEqual(@as(usize, 1), rows.len);
    try expectEqual(@as(DocID, 2), rows[0].doc_id);
}

// ************************************************************************************** Validate
test "validate rejects an uninitialized doc id" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    _ = try inst.put(.{ .doc_id = 0, .start_i = 0, .end_i = 5 }, &.{ 1, 0, 0 });
    try std.testing.expectError(Error.UninitializedDocID, inst.validate());
}

test "validate rejects overlapping ranges within a doc" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    _ = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 10 }, &.{ 1, 0, 0 });
    _ = try inst.put(.{ .doc_id = 1, .start_i = 5, .end_i = 15 }, &.{ 0, 1, 0 });
    try std.testing.expectError(Error.OverlappingVectors, inst.validate());
}

test "validate rejects a vector that is not L2 normalized" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    _ = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 5 }, &.{ 1, 1, 1 });
    try std.testing.expectError(error.NotL2, inst.validate());
}

test "validate ignores removed vectors" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    // Both would fail validation if tombstones were still inspected.
    const bad1 = try inst.put(.{ .doc_id = 0, .start_i = 0, .end_i = 10 }, &.{ 1, 1, 1 });
    const bad2 = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 10 }, &.{ 5, 5, 5 });
    try inst.rm(bad1);
    try inst.rm(bad2);
    _ = try inst.put(.{ .doc_id = 1, .start_i = 0, .end_i = 10 }, &.{ 1, 0, 0 });
    try inst.validate();
}

// ***************************************************************************************** Churn
const ChurnRow = struct {
    doc_id: DocID,
    start_i: usize,
    end_i: usize,
    vec: TestArray,
};

fn verifyAll(inst: *TestStorage, live: *const std.AutoHashMap(VectorID, ChurnRow)) !void {
    try expectEqual(live.count(), inst.len());

    var it = live.iterator();
    while (it.next()) |entry| {
        const id = entry.key_ptr.*;
        const want = entry.value_ptr.*;

        const row = (try inst.get(id)) orelse return error.MissingRow;
        try expectEqual(want.doc_id, row.doc_id);
        try expectEqual(want.start_i, row.start_i);
        try expectEqual(want.end_i, row.end_i);
        try expectVec(inst, id, want.vec);
    }
}

test "churn: multi-cycle reopen with interleaved insert and delete" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    var live = std.AutoHashMap(VectorID, ChurnRow).init(testing_allocator);
    defer live.deinit();

    const cycle1 = [_]ChurnRow{
        .{ .doc_id = 1, .start_i = 0, .end_i = 100, .vec = .{ 1.0, 0.0, 0.0 } },
        .{ .doc_id = 1, .start_i = 100, .end_i = 250, .vec = .{ 0.0, 1.0, 0.0 } },
        .{ .doc_id = 2, .start_i = 0, .end_i = 50, .vec = .{ 0.0, 0.0, 1.0 } },
        .{ .doc_id = 2, .start_i = 50, .end_i = 300, .vec = .{ 0.6, 0.8, 0.0 } },
        .{ .doc_id = 3, .start_i = 0, .end_i = 1000, .vec = .{ -1.0, 0.0, 0.0 } },
        .{ .doc_id = 3, .start_i = 1000, .end_i = 5000, .vec = .{ 0.0, -0.6, 0.8 } },
        .{ .doc_id = 4, .start_i = 0, .end_i = 10, .vec = .{ -0.8, 0.0, 0.6 } },
        .{ .doc_id = 5, .start_i = 0, .end_i = 99999, .vec = .{ 0.0, 0.0, -1.0 } },
    };

    var delete_ids: [4]VectorID = undefined;
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        for (cycle1, 0..) |r, i| {
            const id = try inst.put(.{ .doc_id = r.doc_id, .start_i = r.start_i, .end_i = r.end_i }, &r.vec);
            try live.put(id, r);
            if (i % 2 == 0) delete_ids[i / 2] = id;
        }
        try verifyAll(&inst, &live);
        try inst.validate();
        try inst.flush();
    }

    // --- Cycle 2: reopen, verify, delete scattered entries ---
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        try verifyAll(&inst, &live);
        for (delete_ids) |id| {
            try inst.rm(id);
            _ = live.remove(id);
        }
        try verifyAll(&inst, &live);
        try inst.flush();
    }

    // --- Cycle 3: reopen, verify the holes, insert past them ---
    const cycle3 = [_]ChurnRow{
        .{ .doc_id = 10, .start_i = 500, .end_i = 600, .vec = .{ 0.0, 1.0, 0.0 } },
        .{ .doc_id = 11, .start_i = 0, .end_i = 42, .vec = .{ -1.0, 0.0, 0.0 } },
        .{ .doc_id = 12, .start_i = 42, .end_i = 12345, .vec = .{ 0.6, -0.8, 0.0 } },
        .{ .doc_id = 13, .start_i = 0, .end_i = 1, .vec = .{ 0.0, 0.0, -1.0 } },
    };
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        try verifyAll(&inst, &live);
        for (delete_ids) |hole| try expect((try inst.get(hole)) == null);

        for (cycle3) |r| {
            const id = try inst.put(.{ .doc_id = r.doc_id, .start_i = r.start_i, .end_i = r.end_i }, &r.vec);
            try live.put(id, r);
            // Append-only: a new id never lands on a tombstone.
            for (delete_ids) |hole| try expect(id != hole);
        }
        try verifyAll(&inst, &live);
        try inst.validate();
        try inst.flush();
    }

    // --- Cycle 4: reopen, drop a whole doc ---
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        try verifyAll(&inst, &live);

        try inst.rmByDocId(1);
        var doomed: [8]VectorID = undefined;
        var n: usize = 0;
        var it = live.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.doc_id == 1) {
                doomed[n] = e.key_ptr.*;
                n += 1;
            }
        }
        for (doomed[0..n]) |id| _ = live.remove(id);

        try verifyAll(&inst, &live);
        try inst.flush();
    }

    // --- Cycle 5: reopen, add a batch large enough to cross chunk boundaries ---
    {
        var inst = try open(tmpD.dir);
        defer inst.deinit();
        try verifyAll(&inst, &live);

        for (0..200) |i| {
            const r = ChurnRow{
                .doc_id = @intCast(100 + i),
                .start_i = i * 200,
                .end_i = (i + 1) * 200,
                .vec = .{ 1, 0, 0 },
            };
            const id = try inst.put(.{ .doc_id = r.doc_id, .start_i = r.start_i, .end_i = r.end_i }, &r.vec);
            try live.put(id, r);
        }
        try verifyAll(&inst, &live);
        try inst.validate();
        try inst.flush();
    }

    // --- Final: nothing was lost ---
    var inst = try open(tmpD.dir);
    defer inst.deinit();
    try verifyAll(&inst, &live);
    try inst.validate();
}

// *********************************************************************************** Concurrency
const Hammer = struct {
    inst: *TestStorage,
    base: usize,

    fn run(self: Hammer) void {
        var results: [8]TestStorage.SearchEntry = undefined;
        for (0..25) |i| {
            const doc: DocID = @intCast(self.base * 25 + i + 1);
            const id = self.inst.put(
                .{ .doc_id = doc, .start_i = 0, .end_i = 10 },
                &.{ 1, 0, 0 },
            ) catch unreachable;
            const row = (self.inst.get(id) catch unreachable).?;
            std.debug.assert(row.doc_id == doc);
            _ = self.inst.search(&.{ 1, 0, 0 }, &results, 0.5) catch unreachable;
        }
    }
};

test "concurrent puts, gets and searches stay consistent" {
    var tmpD = tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var inst = try open(tmpD.dir);
    defer inst.deinit();

    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, Hammer.run, .{Hammer{ .inst = &inst, .base = i }});
    }
    for (threads) |t| t.join();

    try expectEqual(@as(usize, 100), inst.len());

    // Every doc id was written exactly once, by exactly one thread.
    for (1..101) |doc| {
        const rows = try inst.vecsForDoc(testing_allocator, @intCast(doc));
        defer testing_allocator.free(rows);
        try expectEqual(@as(usize, 1), rows.len);
    }
}

// ************************************************************************* Cosine similarity math
test "cosine orthogonal" {
    const a = TestVecType{ 1, 0, 0 };
    const b = TestVecType{ 0, 1, 0 };
    const c = TestVecType{ 0, 0, 1 };

    try expect(cosine_similarity(TestN, TestT, a, b) == 0);
    try expect(cosine_similarity(TestN, TestT, a, c) == 0);
    try expect(cosine_similarity(TestN, TestT, b, c) == 0);
}

test "cosine equal" {
    const a = TestVecType{ 1, 0, 0 };
    try expect(cosine_similarity(TestN, TestT, a, a) == 1);
}

test "cosine reverse" {
    const a = TestVecType{ 1, 0, 0 };
    const b = TestVecType{ -1, 0, 0 };
    try expect(cosine_similarity(TestN, TestT, a, b) == -1);
}

test "cosine 45-degree" {
    const a = TestVecType{ 1, 0, 0 };
    const b = TestVecType{ 1, 1, 0 };
    try expect(cosine_similarity(TestN, TestT, a, b) == 0.70710677);
}

test "cosine similar" {
    const a = TestVecType{ 1, 2, 3 };
    const b = TestVecType{ 1, 1, 1 };
    try expect(cosine_similarity(TestN, TestT, a, b) == 0.9258201);
}

test "cosine zero-vec" {
    const a = TestVecType{ 0, 0, 0 };
    const b = TestVecType{ 1, 0, 0 };
    try expect(cosine_similarity(TestN, TestT, a, b) == 0);
}

const std = @import("std");
const assert = std.debug.assert;
const native_endian = @import("builtin").cpu.arch.endian();

const tmpDir = std.testing.tmpDir;
const testing_allocator = std.testing.allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const tracy = @import("tracy");

const pfile = @import("pfile.zig");
const types = @import("types.zig");
const storedDot = @import("vec_util.zig").storedDot;
const validateL2 = @import("vec_util.zig").validateL2;
const VectorID = types.VectorID;

/// Renamed from NoteID: the store deals in documents, whatever the layer above calls them.
/// `note_id_map.zig` keeps the old name for now.
pub const DocID = @import("note_id_map.zig").NoteID;
