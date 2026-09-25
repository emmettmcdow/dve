const MAX_NOTE_LEN: usize = std.math.maxInt(u32);

pub const Error = error{
    NotQueuedShuttingDown,
    InvalidPath,
    /// The resident code index and the store disagree about how many vectors are live. Every
    /// mutation has to touch both, so this means one path updated only one of them -- which
    /// otherwise shows up as results that silently stop appearing.
    IndexOutOfSync,
};

pub const SearchResult = struct {
    /// The path or the key associated with this vector.
    path: []const u8,
    /// If a document has multiple vectors, then start_i is the index where this vector starts.
    start_i: usize,
    /// If a document has multiple vectors, then start_i is the index where this vector ends.
    end_i: usize,
    /// This is the dot product similarity between the query and the specified vector.
    similarity: f32 = 0.0,
};

/// Optional overrides for the files an embedding model loads at startup. Both fields are
/// absolute paths; a null field leaves the embedder's own resolution (bundle resource, then
/// exe-relative) in place. Embedding models that ship no files of their own -- currently
/// `.apple_nlembedding` -- ignore these.
pub const InitOptions = struct {
    /// Absolute path to the tokenizer file.
    tokenizer_path: ?[]const u8 = null,
    /// Absolute path to the model file or package directory.
    model_path: ?[]const u8 = null,
    /// Candidates stage one hands to stage two, which is also the number of vectors read from
    /// disk per query. Measured on real embeddings, K=100 recovers 0.957 of the exact top-10
    /// and K=500 recovers 0.996; see `experiments/results/binrecall.md`.
    ///
    /// **Raising this much past 1,000 costs more than it looks.** The candidate selection in
    /// `codes.zig` is an insertion sort, so it is quadratic in K: a pure in-memory scan is
    /// 10 ms at K=500 and 150 ms at K=4,000. The disk reads are the smaller half.
    candidates: usize = 500,
    /// Threads for the code scan. Null lets the pool size itself. Four saturate memory
    /// bandwidth on the machine this was measured on; see `experiments/results/hamscan.md`.
    scan_threads: ?usize = null,
};

const BaseEmbedder = union(EmbeddingModel) {
    apple_nlembedding: *NLEmbedder,
    mpnet_embedding: *MpnetEmbedder,
};

/// VectorEngine is the primary way to use the vector engine. It requires selecting an
/// EmbeddingModel. The EmbeddingModel must match the Embedder passed into the initialization func.
pub fn VectorEngine(embedding_model: EmbeddingModel) type {
    const VEC_SZ = embedding_model.vecSize();
    const VEC_TYPE = switch (embedding_model) {
        .apple_nlembedding => NLEmbedder.VEC_TYPE,
        .mpnet_embedding => MpnetEmbedder.VEC_TYPE,
    };

    const EmbedJob = struct {
        path: []const u8,
        contents: []const u8,

        pub fn id(self: @This()) u64 {
            return std.hash.Wyhash.hash(0, self.path);
        }
    };

    // Quantization narrows the element type of a vector, not its length, so only the
    // element type varies with `config.quant`.
    const STORED_VEC_TYPE = switch (config.quant) {
        .none => VEC_TYPE,
        .f_16 => f16,
        .i_8 => i8,
    };
    const RawVector = @Vector(VEC_SZ, VEC_TYPE);
    const StoredVector = @Vector(VEC_SZ, STORED_VEC_TYPE);
    // Array rather than @Vector for anything held in a collection: a @Vector(768, f32) is
    // padded to 4096 bytes and aligned to 4096, so inlining one into a struct costs 8KB an
    // entry. The array form is the bare 3072 bytes, and coerces to StoredVector on use.
    const StoredArray = [VEC_SZ]STORED_VEC_TYPE;

    return struct {
        const Self = @This();
        /// The embedder's output form, which is what `rawVectorSearch` takes.
        pub const Raw = RawVector;
        pub const VecStorage = vstore.VStore(VEC_SZ, STORED_VEC_TYPE);
        /// Stage one: a 1-bit code per vector, resident and scanned linearly. Keyed by the
        /// store's own slot number, which is the entire interface between the two -- neither
        /// module imports the other.
        pub const VecCodes = codes.Codes(VEC_SZ, STORED_VEC_TYPE, codes.DEFAULT_BITS);
        pub const quant = config.quant;

        /// Converts a vector as the embedder produced it into the form VecStorage holds.
        /// The switch is on a comptime config value, so only the configured prong is
        /// analyzed -- a `.none` build never reaches the quantizer.
        fn toStored(v: RawVector) StoredVector {
            return switch (config.quant) {
                .none => v,
                .f_16 => quant32to16(VEC_SZ, v),
                .i_8 => quant32toi8(VEC_SZ, v),
            };
        }

        const WorkQueue = UniqueCircularBuffer(EmbedJob, u64, EmbedJob.id);

        base_embedder: BaseEmbedder,
        embedder: embed.Embedder,
        vec_storage: VecStorage,
        codes: VecCodes,
        /// How many candidates stage one produces per query.
        candidates: usize,
        note_id_map: *NoteIdMap,
        basedir: std.fs.Dir,
        allocator: std.mem.Allocator,
        work_queue: *WorkQueue,
        work_queue_thread: Thread,
        work_queue_mutex: Thread.Mutex = .{},
        work_queue_condition: Thread.Condition = .{},
        work_queue_running: bool,

        /// **`allocator` must be thread-safe.** The engine runs a background embedding thread
        /// from the moment `init` returns, and both it and the caller's thread allocate from
        /// this allocator -- the worker for each document it embeds and for growing the code
        /// index, the caller for every search. A `GeneralPurposeAllocator` or `c_allocator` is
        /// fine; an `ArenaAllocator` is not, and passing one corrupts its free list under
        /// concurrent use rather than failing cleanly. This was implicit for as long as the
        /// work queue has existed and is written down because the tests got it wrong.
        pub fn init(
            allocator: std.mem.Allocator,
            basedir: std.fs.Dir,
            opts: InitOptions,
        ) !*Self {
            const base_embedder = o: switch (embedding_model) {
                .apple_nlembedding => {
                    var e = try allocator.create(NLEmbedder);
                    errdefer allocator.destroy(e);
                    try e.init_self();
                    break :o BaseEmbedder{ .apple_nlembedding = e };
                },
                .mpnet_embedding => {
                    var e = try allocator.create(MpnetEmbedder);
                    errdefer allocator.destroy(e);
                    try e.init_self(.{
                        .absolute_model_path = opts.model_path,
                        .absolute_tokenizer_path = opts.tokenizer_path,
                    });
                    break :o BaseEmbedder{ .mpnet_embedding = e };
                },
            };
            const embedder = switch (embedding_model) {
                .apple_nlembedding => base_embedder.apple_nlembedding.embedder(),
                .mpnet_embedding => base_embedder.mpnet_embedding.embedder(),
            };

            var vecs = try VecStorage.init(allocator, basedir, .{ .path = embedder.path });
            errdefer vecs.deinit();

            // Loaded if a usable file is sitting there, rebuilt from the store if not.
            // Rebuilding reads every vector: 279 ms at 247k measured, but the store is ~143 GB
            // at the full corpus and no page cache holds that, so it becomes ~22 s of cold
            // sequential read at every launch against ~0.3 s to read the codes back.
            var codes_name_buf: [256]u8 = undefined;
            const codes_name = try codesPath(&codes_name_buf, embedder.path);
            var vcodes = (try VecCodes.load(
                allocator,
                basedir,
                codes_name,
                stampOf(&vecs),
                .{ .threads = opts.scan_threads },
            )) orelse blk: {
                var c = try VecCodes.init(allocator, .{
                    .capacity = vecs.slot_n,
                    .threads = opts.scan_threads,
                });
                errdefer c.deinit();
                try buildCodes(&vecs, &c);
                break :blk c;
            };
            errdefer vcodes.deinit();

            const wq = try WorkQueue.init(allocator, 1024);

            const note_id_map = try allocator.create(NoteIdMap);
            errdefer allocator.destroy(note_id_map);
            note_id_map.* = try NoteIdMap.init(allocator, basedir);

            const self = try allocator.create(Self);
            self.* = .{
                .base_embedder = base_embedder,
                .embedder = embedder,
                .vec_storage = vecs,
                .codes = vcodes,
                .candidates = @max(opts.candidates, 1),
                .note_id_map = note_id_map,
                .basedir = basedir,
                .allocator = allocator,
                .work_queue = wq,
                .work_queue_thread = undefined,
                .work_queue_running = true,
            };
            // Spawned *after* the struct is written, not inside the initializer. The worker's
            // first act is to read `self.work_queue` and take `self.work_queue_mutex`, so
            // starting it while the fields it needs are still uninitialized is a race it
            // usually wins and occasionally does not -- it surfaced as a queued document
            // embedding itself out of garbage offsets.
            self.work_queue_thread = try spawn(.{}, Self.workQueueRun, .{self});
            return self;
        }
        pub fn deinit(self: *Self) void {
            if (self.work_queue_running) {
                self.shutdown();
            }
            // Best effort: the index is a cache, and failing to write it costs a rebuild at
            // the next open rather than any correctness. A caller that wants to know calls
            // `saveIndex` itself.
            self.saveIndex() catch |e| {
                std.log.warn("could not save the code index: {t}", .{e});
            };
            self.codes.deinit();
            self.vec_storage.deinit();
            self.embedder.deinit();
            switch (embedding_model) {
                .apple_nlembedding => self.allocator.destroy(self.base_embedder.apple_nlembedding),
                .mpnet_embedding => self.allocator.destroy(self.base_embedder.mpnet_embedding),
            }
            self.work_queue.deinit();
            self.note_id_map.deinit();
            self.allocator.destroy(self.note_id_map);
            self.allocator.destroy(self);
        }
        /// Stops the background work queue from running without de-initializing memory.
        pub fn shutdown(self: *Self) void {
            {
                self.work_queue_mutex.lock();
                defer self.work_queue_mutex.unlock();
                self.work_queue_running = false;
            }
            self.work_queue_condition.signal();
            self.work_queue_thread.join();
        }

        /// Reads every live vector once and encodes it. The store hands back the slot along
        /// with the bytes, so the codes array ends up mirroring the store's slot numbering
        /// without either side having to agree on anything else.
        fn buildCodes(store: *VecStorage, out: *VecCodes) !void {
            const zone = tracy.beginZone(@src(), .{ .name = "vector.zig:buildCodes" });
            defer zone.end();
            var it = try store.iterate();
            defer it.deinit();
            while (try it.next()) |e| try out.put(e.slot, e.vec);
        }

        /// One scored result from stage two: a row and its exact cosine.
        const Scored = struct {
            row: VecStorage.Row,
            similarity: f32,
        };

        /// The whole search, both stages.
        ///
        /// Stage one scans the resident codes and returns `self.candidates` slots by Hamming
        /// distance -- no disk at all. Stage two reads those slots' full vectors and scores
        /// them by exact cosine, which is what fixes the ordering: Hamming over a 384-bit code
        /// estimates the angle with a spread of about 13 bits, so it is a good filter and a
        /// poor ranking. Reranking cannot recover a vector stage one never surfaced, which is
        /// why `candidates` is the quality knob and not just a cost one.
        fn twoStage(
            self: *Self,
            allocator: std.mem.Allocator,
            query: *const StoredArray,
            want: usize,
        ) ![]Scored {
            const zone = tracy.beginZone(@src(), .{ .name = "vector.zig:twoStage" });
            defer zone.end();
            if (want == 0) return &.{};

            const cands = try allocator.alloc(VecCodes.Candidate, self.candidates);
            defer allocator.free(cands);
            const n_cand = try self.codes.search(query, cands);

            var out: std.ArrayList(Scored) = .{};
            errdefer out.deinit(allocator);
            try out.ensureTotalCapacity(allocator, @min(want, n_cand));

            var vec: StoredArray = undefined;
            for (cands[0..n_cand]) |cand| {
                // Null means the index named a slot the store has since freed. That is not an
                // error: the two can drift by one removal and the store is the authority.
                const row = (try self.vec_storage.getSlot(cand.slot, &vec)) orelse continue;
                const sim = storedDotAt(VEC_SZ, STORED_VEC_TYPE, &vec, query);
                if (sim <= self.embedder.threshold) continue;
                insertScored(&out, allocator, want, .{ .row = row, .similarity = sim }) catch |e| return e;
            }
            return out.toOwnedSlice(allocator);
        }

        /// Keeps `out` sorted by descending similarity and no longer than `want`. Insertion
        /// rather than sort-at-the-end because `want` is small -- ten or so for a real query --
        /// and most candidates lose to the tenth-best immediately.
        fn insertScored(
            out: *std.ArrayList(Scored),
            allocator: std.mem.Allocator,
            want: usize,
            s: Scored,
        ) !void {
            if (out.items.len == want and s.similarity <= out.items[want - 1].similarity) return;
            if (out.items.len < want) try out.append(allocator, s) else out.items[want - 1] = s;

            var j = out.items.len - 1;
            while (j > 0 and out.items[j - 1].similarity < s.similarity) : (j -= 1) {
                out.items[j] = out.items[j - 1];
            }
            out.items[j] = s;
        }

        /// Searches the vector database. Can return multiple results per path/key.
        pub fn search(self: *Self, raw_query: []const u8, buf: []SearchResult) !usize {
            const zone = tracy.beginZone(@src(), .{ .name = "vector.zig:search" });
            defer zone.end();
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            if (raw_query.len == 0) return 0;

            const max_results = buf.len;

            const query = stripQuery(raw_query);
            if (query.len == 0) return 0;

            const query_vec_union = (try self.embedder.embed(arena.allocator(), query)) orelse {
                return 0;
            };
            const query_vec: StoredArray = toStored(@field(query_vec_union, @tagName(embedding_model)).*);

            debugSearchHeader(query);
            const scored = try self.twoStage(arena.allocator(), &query_vec, max_results);
            var found_n: usize = 0;
            for (scored) |sc| {
                const p = self.note_id_map.getPath(sc.row.doc_id) orelse continue;
                buf[found_n] = SearchResult{
                    .path = p,
                    .start_i = sc.row.start_i,
                    .end_i = sc.row.end_i,
                    .similarity = sc.similarity,
                };
                found_n += 1;
            }

            std.log.info("Found {d} results searching with `{s}`", .{ found_n, query });
            return found_n;
        }

        /// Searches the vector database. Each path/key will be unique in the results.
        pub fn uniqueSearch(self: *Self, raw_query: []const u8, buf: []SearchResult) !usize {
            const zone = tracy.beginZone(@src(), .{ .name = "vector.zig:uniqueSearch" });
            defer zone.end();
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            if (raw_query.len == 0) return 0;

            const query = stripQuery(raw_query);
            if (query.len == 0) return 0;

            const query_vec_union = (try self.embedder.embed(arena.allocator(), query)) orelse return 0;
            const query_vec: StoredArray = toStored(@field(query_vec_union, @tagName(embedding_model)).*);

            debugSearchHeader(query);
            // Over-fetches, because collapsing to one hit per path can discard most of a
            // result set: a long document matching well occupies many of the top rows.
            const scored = try self.twoStage(arena.allocator(), &query_vec, self.candidates);
            const found_n = scored.len;
            var unique_found_n: usize = 0;
            outer: for (scored) |sc| {
                if (unique_found_n >= buf.len) break;
                const path = self.note_id_map.getPath(sc.row.doc_id) orelse continue;
                for (0..unique_found_n) |j| {
                    if (std.mem.eql(u8, buf[j].path, path)) continue :outer;
                }
                buf[unique_found_n] = SearchResult{
                    .path = path,
                    .start_i = sc.row.start_i,
                    .end_i = sc.row.end_i,
                    .similarity = sc.similarity,
                };
                unique_found_n += 1;
            }

            std.log.info(
                "Condensed {d} duplicate results to {d} duplicate searching with {s}\n",
                .{ found_n, unique_found_n, query },
            );
            return unique_found_n;
        }

        /// Given a sentence(result_content), highlight words in the sentence that most match the
        /// query. This can be used to highlight to the user *why* a search result appeared.
        pub fn populateHighlights(
            self: *Self,
            query: []const u8,
            result_content: []const u8,
            highlights: []usize,
        ) !void {
            const zone = tracy.beginZone(@src(), .{ .name = "vector.zig:populateHighlights" });
            defer zone.end();

            const max_highlights = @divExact(highlights.len, 2);
            if (max_highlights == 0) return;

            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            const query_vec_union = (try self.embedder.embed(arena.allocator(), query)) orelse return;
            const query_vec = @field(query_vec_union, @tagName(embedding_model)).*;

            var found: u8 = 0;
            var wordspliterator = embed.WordSpliterator.init(result_content);
            while (wordspliterator.next()) |word_chunk| {
                const chunk_vec_union = (try self.embedder.embed(
                    arena.allocator(),
                    word_chunk.contents,
                )) orelse continue;
                const chunk_vec = @field(chunk_vec_union, @tagName(embedding_model)).*;

                const similar = vstore.cosine_similarity(
                    VEC_SZ,
                    VEC_TYPE,
                    chunk_vec,
                    query_vec,
                );
                if (similar > self.embedder.strict_threshold) {
                    highlights[found * 2] = word_chunk.start_i;
                    highlights[found * 2 + 1] = word_chunk.end_i;
                    found += 1;
                    if (found >= max_highlights) return;
                }
            }
            return;
        }

        fn workQueueRun(self: *@This()) !void {
            var n_embedded: usize = 0;
            const unflushed_limit = 100;
            while (true) {
                self.work_queue_mutex.lock();
                const job = while (true) {
                    if (self.work_queue.pop()) |j| {
                        self.work_queue_mutex.unlock();
                        break j;
                    }
                    if (!self.work_queue_running) {
                        self.work_queue_mutex.unlock();
                        try self.save();
                        return;
                    }
                    self.work_queue_condition.wait(&self.work_queue_mutex);
                };
                defer self.allocator.free(job.contents);
                defer self.allocator.free(job.path);
                self.embedTextInternal(job.path, job.contents) catch |err| {
                    std.log.err("embedText error: {}", .{err});
                };
                n_embedded += 1;
                if (n_embedded >= unflushed_limit) {
                    n_embedded = 0;
                    try self.save();
                }
            }
        }

        const EmbeddedSentence = struct {
            vec: *const StoredArray,
            start_i: usize,
            end_i: usize,
        };

        // Embed an entire document, a collection of sentences. And associate it with a key/path.
        // Runs synchronously.
        pub fn embedText(self: *Self, path: []const u8, contents: []const u8) !void {
            if (path.len == 0) return Error.InvalidPath;
            try self.embedTextInternal(path, contents);
            return self.save();
        }

        /// Persist changes to disk.
        pub fn save(self: *Self) !void {
            return self.vec_storage.flush();
        }

        fn embedTextInternal(self: *Self, path: []const u8, contents: []const u8) !void {
            const zone = tracy.beginZone(@src(), .{ .name = "vector.zig:embedText" });
            defer zone.end();
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();
            const allocator = arena.allocator();
            assert(contents.len < MAX_NOTE_LEN);

            var embedded_sentence_list: std.ArrayList(EmbeddedSentence) = .{};
            errdefer embedded_sentence_list.deinit(allocator);
            var spliterator = embed.SentenceSpliterator.init(contents);
            while (spliterator.next()) |sentence| {
                const vec: ?*const RawVector =
                    if (whitespaceOnly(sentence.contents) or !wordlike(sentence.contents))
                        null
                    else if (try self.embedder.embed(allocator, sentence.contents)) |v|
                        @field(v, @tagName(embedding_model))
                    else
                        null;
                if (vec) |raw_vec| {
                    // Unquantized, the embedder's own arena buffer is already the storage
                    // form, so it is borrowed as-is. Quantizing needs somewhere to put the
                    // narrowed copy; the arena outlives replaceVectors below.
                    const stored: *const StoredArray = if (config.quant == .none)
                        @ptrCast(raw_vec)
                    else stored: {
                        const narrowed = try allocator.create(StoredArray);
                        narrowed.* = toStored(raw_vec.*);
                        break :stored narrowed;
                    };
                    try embedded_sentence_list.append(allocator, .{
                        .vec = stored,
                        .start_i = sentence.start_i,
                        .end_i = sentence.end_i,
                    });
                }
            }
            const embedded_sentences = try embedded_sentence_list.toOwnedSlice(allocator);
            const note_id = try self.note_id_map.getOrCreateId(path);
            try self.replaceVectors(allocator, note_id, embedded_sentences);

            std.log.info("Embedded {d} sentences\n", .{embedded_sentences.len});
        }

        // Embed an entire document, a collection of sentences. And associate it with a key/path.
        // Runs asynchronously.
        pub fn embedTextAsync(self: *Self, path: []const u8, contents: []const u8) !void {
            if (path.len == 0) return Error.InvalidPath;

            const owned_path = try self.allocator.dupe(u8, path);
            errdefer self.allocator.free(owned_path);
            const owned_contents = try self.allocator.dupe(u8, contents);
            errdefer self.allocator.free(owned_contents);

            // The push and the signal both have to happen under work_queue_mutex. The
            // consumer holds that mutex from the pop that comes up empty until it waits on
            // the condition, so a push that lands in between would not wake it.
            self.work_queue_mutex.lock();
            defer self.work_queue_mutex.unlock();
            if (!self.work_queue_running) return Error.NotQueuedShuttingDown;

            // Re-queuing a path replaces the job still sitting in the queue. That job's
            // buffers are ours to free -- nothing else refers to them.
            const displaced = try self.work_queue.push(.{
                .path = owned_path,
                .contents = owned_contents,
            });
            if (displaced) |old| {
                self.allocator.free(old.path);
                self.allocator.free(old.contents);
            }
            self.work_queue_condition.signal();
        }

        fn replaceVectors(
            self: *Self,
            allocator: std.mem.Allocator,
            note_id: NoteID,
            embedded_sentences: []const EmbeddedSentence,
        ) !void {
            const old_vecs = try self.vec_storage.vecsForDoc(allocator, note_id);
            defer allocator.free(old_vecs);

            // Remove before putting, so the new rows land in the slots the old ones vacate.
            // Either order is correct -- a `put` can only claim a slot that is already free, so
            // it can never land on a live vector -- but putting first leaves the old rows live
            // while the new ones are placed, which forces every replaced document to occupy two
            // generations' worth of slots.
            //
            // This order is also the better failure mode. If a `put` below fails partway, the
            // document is left under-indexed, which a re-embed fixes. Putting first and failing
            // partway leaves the old rows *and* some new ones covering the same offsets, which
            // is a state `validate` rejects as overlapping.
            for (old_vecs) |old_v| {
                // Only a double-remove is benign here: the row came from `vecsForDoc`, so
                // anything else -- an IO error, a torn record -- is a real failure and has to
                // reach the caller rather than be swallowed as "already gone".
                self.vec_storage.rm(old_v.vec_id) catch |e| switch (e) {
                    error.MultipleRemove => continue,
                    else => return e,
                };
                self.codes.rm(VecStorage.slotOf(old_v.vec_id));
            }

            for (embedded_sentences) |sentence| {
                const id = try self.vec_storage.put(.{
                    .doc_id = note_id,
                    .start_i = sentence.start_i,
                    .end_i = sentence.end_i,
                }, sentence.vec);
                // Indexing can only fail on allocation, but if it does the vector would be on
                // disk and invisible to every search until the next rebuild. Undoing the store
                // write keeps the two in step; the document is then under-indexed, which a
                // re-embed fixes, and that is the same failure mode as a `put` that fails.
                errdefer self.vec_storage.rm(id) catch {};
                try self.codes.put(VecStorage.slotOf(id), sentence.vec);
            }
        }

        /// The database filename this engine uses, so a test can name the index beside it.
        pub fn embedderPathForTest() []const u8 {
            return switch (embedding_model) {
                .apple_nlembedding => NLEmbedder.PATH,
                .mpnet_embedding => MpnetEmbedder.PATH,
            };
        }

        /// Writes the code index next to the database so the next open does not have to
        /// rebuild it from every vector.
        ///
        /// `deinit` calls this, which means an unclean exit leaves nothing to load and the
        /// next open rebuilds -- deliberately. A saved index cannot be proved fresh from
        /// counters alone (`codes.Stamp` explains why), so it is only trusted when the process
        /// that wrote it also shut down cleanly. Call this explicitly to checkpoint sooner;
        /// it costs one sequential write of the index, ~48 bytes a vector.
        pub fn saveIndex(self: *Self) !void {
            var buf: [256]u8 = undefined;
            const name = try codesPath(&buf, self.embedder.path);
            try self.codes.save(self.basedir, name, stampOf(&self.vec_storage));
        }

        /// What a saved index has to match to be loaded. Read from the store rather than
        /// tracked, so it cannot drift from the thing it describes.
        fn stampOf(store: *VecStorage) codes.Stamp {
            return .{
                .store_bytes = store.file.size() catch 0,
                .slot_n = store.slot_n,
                .vec_n = store.vec_n,
            };
        }

        /// Validate the vector database is in a good state.
        pub fn validate(self: *Self) !void {
            try self.vec_storage.validate();
            // Cheap, and it is the invariant that binds the two halves together: stage one
            // can only return what it was told about, so an index that has drifted from the
            // store loses results with no error anywhere.
            if (self.codes.len() != self.vec_storage.len()) return Error.IndexOutOfSync;
        }

        /// Delete the entries associated with a given path.
        pub fn removePath(self: *Self, path: []const u8) !void {
            if (path.len == 0) return Error.InvalidPath;
            if (self.note_id_map.getId(path)) |note_id| {
                // Collected before the removal, because afterwards there is nothing to look up.
                const rows = try self.vec_storage.vecsForDoc(self.allocator, note_id);
                defer self.allocator.free(rows);
                try self.vec_storage.rmByDocId(note_id);
                for (rows) |row| self.codes.rm(VecStorage.slotOf(row.vec_id));
            }
            try self.note_id_map.removePath(path);
        }

        /// For entries associated with a given path, associate them with a different path.
        pub fn renamePath(self: *Self, old_path: []const u8, new_path: []const u8) !void {
            if (old_path.len == 0 or new_path.len == 0) return Error.InvalidPath;
            try self.note_id_map.renamePath(old_path, new_path);
        }

        fn getPath(self: *Self, note_id: NoteID) ?[]const u8 {
            return self.note_id_map.getPath(note_id);
        }

        /// Remove paths which no longer have a vector associated with them.
        pub fn pruneOrphanedPaths(self: *Self, basedir: std.fs.Dir) !void {
            try self.note_id_map.pruneOrphanedPaths(basedir);
        }

        fn debugSearchHeader(query: []const u8) void {
            if (!config.debug) return;
            std.debug.print("Checking similarity against '{s}':\n", .{query});
        }

        /// Turns query text into the vector `rawVectorSearch` takes, and nothing else.
        ///
        /// Split out so a caller can charge the embedder separately from the search it feeds.
        /// On CoreML an embedding is ~20 ms against ~1 ms of search, so a benchmark that times
        /// `search` end to end is reporting the model, not the index. Null for a query that
        /// strips to nothing or that the embedder declines.
        pub fn embedQuery(self: *Self, raw_query: []const u8) !?RawVector {
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();
            const query = stripQuery(raw_query);
            if (query.len == 0) return null;
            const out = (try self.embedder.embed(arena.allocator(), query)) orelse return null;
            return @field(out, @tagName(embedding_model)).*;
        }

        /// Exhaustive search: scores every vector in the store against the query, with no
        /// index involved. Slow by construction and O(corpus) on every call.
        ///
        /// This is not a path the app should use. It exists as the ground truth a benchmark
        /// measures the two-stage path against -- recall is only meaningful against the answer
        /// an exhaustive scan would have given, and computing that needs the scan.
        pub fn exactVectorSearch(self: *Self, raw_vec: RawVector, buf: []SearchResult) !usize {
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            const vec: StoredArray = toStored(raw_vec);
            const entries = try arena.allocator().alloc(VecStorage.SearchEntry, buf.len);
            const n = try self.vec_storage.search(&vec, entries, self.embedder.threshold);

            var found: usize = 0;
            for (entries[0..n]) |e| {
                const p = self.note_id_map.getPath(e.row.doc_id) orelse continue;
                buf[found] = .{
                    .path = p,
                    .start_i = e.row.start_i,
                    .end_i = e.row.end_i,
                    .similarity = e.similarity,
                };
                found += 1;
            }
            return found;
        }

        /// Searches the vector database with an already-embedded query vector. Behaves like
        /// `search`, but runs no embedding operations.
        pub fn rawVectorSearch(self: *Self, raw_vec: RawVector, buf: []SearchResult) !usize {
            const zone = tracy.beginZone(@src(), .{ .name = "vector.zig:rawVectorSearch" });
            defer zone.end();
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            const vec: StoredArray = toStored(raw_vec);

            const scored = try self.twoStage(arena.allocator(), &vec, buf.len);
            var found_n: usize = 0;
            for (scored) |sc| {
                const p = self.note_id_map.getPath(sc.row.doc_id) orelse continue;
                buf[found_n] = SearchResult{
                    .path = p,
                    .start_i = sc.row.start_i,
                    .end_i = sc.row.end_i,
                    .similarity = sc.similarity,
                };
                found_n += 1;
            }

            std.log.info("Found {d} results searching with raw vector", .{found_n});
            return found_n;
        }
    };
}

/// The index file sits beside the database and is named after it, so a build with a different
/// quantization -- which already gets its own `.db` -- gets its own index too rather than
/// silently reading one built for a different vector type.
fn codesPath(buf: []u8, db_path: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.codes", .{db_path});
}

fn stripQuery(query: []const u8) []const u8 {
    var start_i: usize = 0;
    var end_i = query.len;
    for (query, 0..) |c, i| {
        start_i = i;
        if (isAlphanumeric(c)) {
            break;
        }
    }
    const new_len = end_i - start_i;
    if (start_i == end_i - 1) return query[0..0];
    for (1..new_len + 1) |neg_i| {
        const i = query.len - neg_i;
        const c = query[i];
        if (isAlphanumeric(c)) {
            break;
        }
        end_i = i;
    }
    assert(start_i <= end_i);
    return query[start_i..end_i];
}

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

const TestVecDB = VectorEngine(.apple_nlembedding);
// Follows the build's quantization setting -- tests that read vectors back out of storage
// get them in whatever type they were stored as, not necessarily the embedder's f32.
const TestVector = TestVecDB.VecStorage.Array;
/// The element type `TestVector` is made of, named once so the tests below can compare two
/// stored vectors without restating whatever -Dstorage-quantize picked.
const STORED_VEC_TYPE_T = @typeInfo(TestVector).array.child;
fn getVectorsForPath(db: *TestVecDB, path: []const u8, buf: []TestVector) !usize {
    const note_id = db.note_id_map.getId(path) orelse return 0;
    const vec_rows = try db.vec_storage.vecsForDoc(testing_allocator, note_id);
    defer testing_allocator.free(vec_rows);
    for (vec_rows, 0..) |v, i| {
        try db.vec_storage.getVec(v.vec_id, &buf[i]);
    }
    return vec_rows.len;
}

test "embedText hello" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    const text = "hello";
    try db.embedText(path, text);

    var buf: [1]SearchResult = undefined;
    try expectEqual(1, try db.search(text, &buf));

    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 5 },
    }, buf[0..1]);

    try db.validate();
}

test "embedText skip empties" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    const text = "/hello/";
    try db.embedText(path, text);

    try db.validate();
}

test "embedText clear previous" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "hello");

    var buf: [1]SearchResult = undefined;
    try expectEqual(1, try db.search("hello", &buf));
    try db.embedText(path, "flatiron");
    try expectEqual(0, try db.search("hello", &buf));

    try db.validate();
}

test "embedText re-embedding a document reuses its slots" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    // `replaceVectors` removes the old rows before putting the new ones, so every re-embed
    // refills the slots it just freed and the high-water mark never moves. Putting first would
    // hold both generations live at once, settling at two slots per sentence forever.
    const sentences = 20;
    const text = "pizza. " ** sentences;
    const path = "test.md";

    try db.embedText(path, text);
    try expectEqual(sentences, db.vec_storage.vec_n);
    try expectEqual(sentences, db.vec_storage.slot_n);

    for (0..5) |_| {
        try db.embedText(path, text);
        try expectEqual(sentences, db.vec_storage.vec_n);
        try expectEqual(sentences, db.vec_storage.slot_n);
    }
    try db.validate();
}

test "search" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "pizza. pizza. pizza.");

    var buffer: [10]SearchResult = undefined;
    try expectEqual(3, try db.search("pizza", &buffer));
    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 5 },
        .{ .path = path, .start_i = 7, .end_i = 12 },
        .{ .path = path, .start_i = 14, .end_i = 19 },
    }, buffer[0..3]);

    try db.validate();
}

test "search mpnet" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try VectorEngine(.mpnet_embedding).init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "pizza. pizza. pizza.");

    var buffer: [10]SearchResult = undefined;
    try expectEqual(3, try db.search("pizza", &buffer));
    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 5 },
        .{ .path = path, .start_i = 7, .end_i = 12 },
        .{ .path = path, .start_i = 14, .end_i = 19 },
    }, buffer[0..3]);

    try db.validate();
}

// Verifies the `opts` argument reaches the embedder: the files it loads are the ones named
// here, not the ones it would have resolved on its own.
test "init opts model paths" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Under test the default model files live in <cwd>/zig-out/share.
    const cwd = try std.fs.cwd().realpathAlloc(alloc, ".");
    const model_path = try std.fmt.allocPrint(
        alloc,
        "{s}/zig-out/{s}",
        .{ cwd, MpnetEmbedder.MODEL_PATH },
    );

    // A copy of the tokenizer somewhere the default resolution would never look, so the
    // assertion below fails if opts is dropped on the way to the embedder.
    try std.fs.cwd().copyFile(
        try std.fmt.allocPrint(alloc, "{s}/zig-out/{s}", .{ cwd, MpnetEmbedder.TOKENIZER_PATH }),
        tmpD.dir,
        "custom_tokenizer.json",
        .{},
    );
    const tokenizer_path = try tmpD.dir.realpathAlloc(alloc, "custom_tokenizer.json");

    var db = try VectorEngine(.mpnet_embedding).init(alloc, tmpD.dir, .{
        .model_path = model_path,
        .tokenizer_path = tokenizer_path,
    });
    defer db.deinit();

    const mpnet = db.base_embedder.mpnet_embedding;
    try expectEqualSlices(u8, model_path, mpnet.loaded_model_path);
    try expectEqualSlices(u8, tokenizer_path, mpnet.loaded_tokenizer_path);

    // And the engine built from those files works.
    const path = "test.md";
    try db.embedText(path, "pizza. pizza. pizza.");
    var buffer: [10]SearchResult = undefined;
    try expectEqual(3, try db.search("pizza", &buffer));
    try db.validate();
}

test "uniqueSearch" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "pizza. pizza. pizza.");

    var buffer: [10]SearchResult = undefined;
    try expectEqual(1, try db.uniqueSearch("pizza", &buffer));
    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 5 },
    }, buffer[0..1]);

    try db.validate();
}

test "search returns results with similarity" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "brick. tacos. pizza.");
    db.embedder.threshold = 0.0;

    var buffer: [10]SearchResult = undefined;
    try expectEqual(3, try db.search("pizza", &buffer));

    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path, .start_i = 14, .end_i = 19 },
        .{ .path = path, .start_i = 7, .end_i = 12 },
        .{ .path = path, .start_i = 0, .end_i = 5 },
    }, buffer[0..3]);

    try std.testing.expect(buffer[0].similarity > 0);
    try std.testing.expect(buffer[1].similarity > 0);
    try std.testing.expect(buffer[2].similarity > 0);
    try std.testing.expect(buffer[0].similarity >= buffer[1].similarity);
    try std.testing.expect(buffer[1].similarity >= buffer[2].similarity);

    try db.validate();
}

test "uniqueSearch returns results with similarity" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path1 = "test1.md";
    try db.embedText(path1, "brick");
    const path2 = "test2.md";
    try db.embedText(path2, "tacos");
    const path3 = "test3.md";
    try db.embedText(path3, "pizza");
    db.embedder.threshold = 0.0;

    var buffer: [10]SearchResult = undefined;
    try expectEqual(3, try db.search("pizza", &buffer));

    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path3, .start_i = 0, .end_i = 5 },
        .{ .path = path2, .start_i = 0, .end_i = 5 },
        .{ .path = path1, .start_i = 0, .end_i = 5 },
    }, buffer[0..3]);

    try std.testing.expect(buffer[0].similarity > 0);
    try std.testing.expect(buffer[1].similarity > 0);
    try std.testing.expect(buffer[2].similarity > 0);
    try std.testing.expect(buffer[0].similarity >= buffer[1].similarity);
    try std.testing.expect(buffer[1].similarity >= buffer[2].similarity);

    try db.validate();
}

fn RawVec(comptime model: EmbeddingModel) type {
    return switch (model) {
        .apple_nlembedding => @Vector(NLEmbedder.VEC_SZ, NLEmbedder.VEC_TYPE),
        .mpnet_embedding => @Vector(MpnetEmbedder.VEC_SZ, MpnetEmbedder.VEC_TYPE),
    };
}

/// Embeds `text` outside of the engine, so tests can hand rawVectorSearch a query vector
/// without the engine doing any embedding itself.
fn rawQueryVec(
    comptime model: EmbeddingModel,
    e: *embed.Embedder,
    allocator: std.mem.Allocator,
    text: []const u8,
) !RawVec(model) {
    const v = (try e.embed(allocator, text)) orelse return error.TestEmbedFailed;
    return @field(v, @tagName(model)).*;
}

test "rawVectorSearch hello" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "hello");

    const query = try rawQueryVec(.apple_nlembedding, &db.embedder, arena.allocator(), "hello");
    var buf: [1]SearchResult = undefined;
    try expectEqual(1, try db.rawVectorSearch(query, &buf));
    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 5 },
    }, buf[0..1]);
    try expect(buf[0].similarity > 0);

    try db.validate();
}

test "rawVectorSearch matches search" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "pizza. pizza. pizza.");

    var expected: [10]SearchResult = undefined;
    const expected_n = try db.search("pizza", &expected);
    try expectEqual(3, expected_n);

    const query = try rawQueryVec(.apple_nlembedding, &db.embedder, arena.allocator(), "pizza");
    var actual: [10]SearchResult = undefined;
    try expectEqual(expected_n, try db.rawVectorSearch(query, &actual));
    try expectSearchResultsIgnoresimilarity(expected[0..expected_n], actual[0..expected_n]);
    for (expected[0..expected_n], actual[0..expected_n]) |e, a| {
        try expectEqual(e.similarity, a.similarity);
    }

    try db.validate();
}

test "rawVectorSearch returns results with similarity" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "brick. tacos. pizza.");
    db.embedder.threshold = 0.0;

    const query = try rawQueryVec(.apple_nlembedding, &db.embedder, arena.allocator(), "pizza");
    var buffer: [10]SearchResult = undefined;
    try expectEqual(3, try db.rawVectorSearch(query, &buffer));

    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path, .start_i = 14, .end_i = 19 },
        .{ .path = path, .start_i = 7, .end_i = 12 },
        .{ .path = path, .start_i = 0, .end_i = 5 },
    }, buffer[0..3]);

    try expect(buffer[0].similarity > 0);
    try expect(buffer[0].similarity >= buffer[1].similarity);
    try expect(buffer[1].similarity >= buffer[2].similarity);

    try db.validate();
}

test "rawVectorSearch no matches" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    try db.embedText("test.md", "pizza");

    // A zero vector has a zero dot product against everything, so nothing clears the threshold.
    const query: TestVector = @splat(0.0);
    var buffer: [10]SearchResult = undefined;
    try expectEqual(0, try db.rawVectorSearch(query, &buffer));

    try db.validate();
}

test "rawVectorSearch empty database" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const query = try rawQueryVec(.apple_nlembedding, &db.embedder, arena.allocator(), "pizza");
    var buffer: [10]SearchResult = undefined;
    try expectEqual(0, try db.rawVectorSearch(query, &buffer));
}

test "rawVectorSearch cap results" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    for (0..5) |i| {
        var path_buf: [16]u8 = undefined;
        try db.embedText(try bufPrint(&path_buf, "test{d}.md", .{i}), "brick");
    }

    const query = try rawQueryVec(.apple_nlembedding, &db.embedder, arena.allocator(), "brick");
    var buffer: [2]SearchResult = undefined;
    try expectEqual(2, try db.rawVectorSearch(query, &buffer));
    try expectEqual(0, try db.rawVectorSearch(query, &.{}));

    try db.validate();
}

test "rawVectorSearch skips removed paths" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "pizza");

    const query = try rawQueryVec(.apple_nlembedding, &db.embedder, arena.allocator(), "pizza");
    var buffer: [10]SearchResult = undefined;
    try expectEqual(1, try db.rawVectorSearch(query, &buffer));

    try db.removePath(path);
    try expectEqual(0, try db.rawVectorSearch(query, &buffer));

    try db.validate();
}

test "rawVectorSearch mpnet" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try VectorEngine(.mpnet_embedding).init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "pizza. pizza. pizza.");

    const query = try rawQueryVec(.mpnet_embedding, &db.embedder, arena.allocator(), "pizza");
    var buffer: [10]SearchResult = undefined;
    try expectEqual(3, try db.rawVectorSearch(query, &buffer));
    try expectSearchResultsIgnoresimilarity(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 5 },
        .{ .path = path, .start_i = 7, .end_i = 12 },
        .{ .path = path, .start_i = 14, .end_i = 19 },
    }, buffer[0..3]);

    try db.validate();
}

test "embed chunk cleanup" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    // Make the threshold strict, results should be exact matches.
    db.embedder.threshold = 0.95;

    const cases = [_]struct { name: []const u8, query: []const u8, entry: []const u8 }{
        .{
            .name = "link-removal",
            .query = "dogs",
            .entry = "dogs https://en.wikipedia.org/wiki/Dog",
        },
        .{
            .name = "space-removal",
            .query = "dogs",
            .entry = "        dogs          ",
        },
        .{
            .name = "h1-removal",
            .query = "dogs",
            .entry = "# dogs",
        },
        .{
            .name = "hN-removal",
            .query = "dogs",
            .entry = "###### dogs",
        },
        .{
            .name = "garbage-removal",
            .query = "dogs",
            .entry = "@ #$%^&*()_+<>:\"{}|,/;'[]\\dogs@ #$%^&*()_+<>:\"{}|,/;'[]\\",
        },
    };
    for (cases) |case| {
        try db.embedText(case.name, case.entry);
        defer db.removePath(case.name) catch @panic("this should not happen!");
        var result: [1]SearchResult = undefined;
        try expectEqualCase(1, try db.search(case.query, &result), case.name);
        try expectEqualCase(1, try db.uniqueSearch(case.query, &result), case.name);
    }
}

fn expectEqualCase(a: anytype, b: anytype, case: []const u8) !void {
    expectEqual(a, b) catch |e| {
        std.debug.print("... on case: {s}\n", .{case});
        return e;
    };
}

test "search cap results" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    for (0..150) |i| {
        var buf: [10]u8 = undefined;
        try db.embedText(try bufPrint(&buf, "{d}", .{i}), "brick");
    }
    var results: [100]SearchResult = undefined;
    try expectEqual(100, try db.search("brick", &results));
    try expectEqual(100, try db.uniqueSearch("brick", &results));
}

test "search strip queries" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    var results: [1]SearchResult = undefined;
    _ = try db.search("??foo??", &results);
    _ = try db.uniqueSearch("??foo??", &results);
}

fn expectSearchResultsIgnoresimilarity(expected: []const SearchResult, actual: []const SearchResult) !void {
    if (expected.len != actual.len) {
        std.debug.print(
            "slice lengths differ: expected {d}, found {d}\n",
            .{ expected.len, actual.len },
        );
        return error.TestExpectedEqual;
    }
    for (expected, actual, 0..) |e, a, i| {
        if (!std.mem.eql(u8, e.path, a.path) or e.start_i != a.start_i or e.end_i != a.end_i) {
            std.debug.print(
                "index {d}: expected {{ .path = {s}, .start_i = {d}, .end_i = {d} }}, found {{ .path = {s}, .start_i = {d}, .end_i = {d} }}\n",
                .{ i, e.path, e.start_i, e.end_i, a.path, a.start_i, a.end_i },
            );
            return error.TestExpectedEqual;
        }
    }
}

pub fn testEmbedder(allocator: std.mem.Allocator) !struct { e: *NLEmbedder, iface: embed.Embedder } {
    const e = try allocator.create(NLEmbedder);
    e.* = try NLEmbedder.init();
    return .{ .e = e, .iface = e.embedder() };
}

test "embedText same input same result" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "apple");
    var initial_vecs: [1]TestVector = undefined;
    try expectEqual(1, try getVectorsForPath(db, path, &initial_vecs));

    try db.embedText(path, "apple");
    var updated_vecs: [1]TestVector = undefined;
    try expectEqual(1, try getVectorsForPath(db, path, &updated_vecs));

    try std.testing.expect(std.mem.eql(STORED_VEC_TYPE_T, &initial_vecs[0], &updated_vecs[0]));

    try db.validate();
}

test "embedText different input different result" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "apple");
    var initial_vecs: [1]TestVector = undefined;
    try expectEqual(1, try getVectorsForPath(db, path, &initial_vecs));

    try db.embedText(path, "banana");
    var updated_vecs: [1]TestVector = undefined;
    try expectEqual(1, try getVectorsForPath(db, path, &updated_vecs));

    // Vector should be different (apple != banana)
    try std.testing.expect(!std.mem.eql(STORED_VEC_TYPE_T, &initial_vecs[0], &updated_vecs[0]));

    try db.validate();
}

test "embedText updates only changed sentences" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    // Initial content: three one-word sentences (only embeddable words get stored)
    const initial_content = "apple. banana. cherry.";
    try db.embedText(path, initial_content);

    var initial_vecs: [3]TestVector = undefined;
    try expectEqual(3, try getVectorsForPath(db, path, &initial_vecs));

    // Updated content: same first and last words, different middle word
    const updated_content = "apple. dragonfruit. cherry.";
    try db.embedText(path, updated_content);

    var updated_vecs: [3]TestVector = undefined;
    try expectEqual(3, try getVectorsForPath(db, path, &updated_vecs));

    try std.testing.expect(std.mem.eql(STORED_VEC_TYPE_T, &initial_vecs[0], &updated_vecs[0]));
    try std.testing.expect(!std.mem.eql(STORED_VEC_TYPE_T, &initial_vecs[1], &updated_vecs[1]));
    try std.testing.expect(std.mem.eql(STORED_VEC_TYPE_T, &initial_vecs[2], &updated_vecs[2]));

    try db.validate();
}

test "embedText handle multiple remove gracefully" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    const initial_content = "foo.\nfoo.\nfoo.";
    const updated_content = "bar.\nbar.\nbar.";
    try db.embedText(path, "");
    try db.embedText(path, initial_content);
    try db.embedText(path, updated_content);

    try db.validate();
}

test "populateHighlights" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    {
        const query = "hello";
        const contents = "bah hello";
        var highlights: [10]usize = .{0} ** 10;
        try db.populateHighlights(query, contents, &highlights);
        try expectEqualSlices(usize, &[10]usize{ 4, 9, 0, 0, 0, 0, 0, 0, 0, 0 }, &highlights);
    }
    { // Multiple hits
        const query = "hello";
        const contents = "hello; hello ";
        var highlights: [10]usize = .{0} ** 10;
        try db.populateHighlights(query, contents, &highlights);
        try expectEqualSlices(usize, &[10]usize{ 0, 5, 7, 12, 0, 0, 0, 0, 0, 0 }, &highlights);
    }

    try db.validate();
}

test "embed skip low-value" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try VectorEngine(.mpnet_embedding).init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();
    db.embedder.threshold = 0;

    {
        const query = " ";
        const contents = " ";
        const path = "test1.md";
        try db.embedText(path, contents);
        var buffer: [10]SearchResult = undefined;
        try expectEqual(0, try db.search(query, &buffer));
    }
    {
        const query = " ";
        const contents = "  ";
        const path = "test2.md";
        try db.embedText(path, contents);
        var buffer: [10]SearchResult = undefined;
        try expectEqual(0, try db.search(query, &buffer));
    }
    {
        const query = " ";
        const contents = " \n\r\t ";
        const path = "test3.md";
        try db.embedText(path, contents);
        var buffer: [10]SearchResult = undefined;
        try expectEqual(0, try db.search(query, &buffer));
    }
    {
        const query = "a";
        const contents = "a#";
        const path = "test4.md";
        try db.embedText(path, contents);
        var buffer: [10]SearchResult = undefined;
        try expectEqual(0, try db.search(query, &buffer));
    }
    try db.validate();
}

test "embedTextAsync" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path1 = "test1.md";
    const path2 = "test2.md";
    const path3 = "test3.md";

    try db.embedTextAsync(path1, "pizza");
    try db.embedTextAsync(path2, "pizza");
    try db.embedTextAsync(path3, "pizza");

    std.Thread.sleep(2 * std.time.ns_per_s);

    var buffer: [10]SearchResult = undefined;
    const found = try db.search("pizza", &buffer);
    try expectEqual(3, found);

    try db.validate();
}

test "embedTextAsync drains queue on shutdown" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const N = 60;

    for (0..N) |i| {
        var path_buf: [16]u8 = undefined;
        const path = bufPrint(&path_buf, "test{d}.md", .{i + 1}) catch unreachable;
        try db.embedTextAsync(path, "pizza");
    }
    db.shutdown();

    var buffer: [N]SearchResult = undefined;
    const found = try db.search("pizza", &buffer);
    try expectEqual(N, found);

    try db.validate();
}

// Uses testing_allocator rather than an arena so that a job dropped without being freed
// is reported. Re-queuing a path replaces the job already in the queue, and the replaced
// job's path/contents belong to nobody but us.
test "embedTextAsync frees jobs it replaces in the queue" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const N = 60;

    // Two passes over the same paths. The queue holds 63 and the embedder is far slower
    // than these pushes, so the second pass lands on jobs that are still queued.
    for (0..2) |_| {
        for (0..N) |i| {
            var path_buf: [16]u8 = undefined;
            const path = bufPrint(&path_buf, "test{d}.md", .{i + 1}) catch unreachable;
            while (true) {
                db.embedTextAsync(path, "pizza") catch |err| switch (err) {
                    error.Full => continue,
                    else => return err,
                };
                break;
            }
        }
    }
    db.shutdown();

    // A replaced job must be embedded once, not twice: N paths, one vector each.
    var buffer: [N * 2]SearchResult = undefined;
    try expectEqual(N, try db.search("pizza", &buffer));

    try db.validate();
}

test "embedTextAsync rejects after shutdown" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    db.shutdown();
    try expectEqual(Error.NotQueuedShuttingDown, db.embedTextAsync(path, "pizza"));
}

/// A second engine opened over the same directory. Tests use it to assert on what actually
/// landed on disk instead of on the in-memory state of the engine that wrote it.
const Reopened = struct {
    db: *TestVecDB,
    embedder: *NLEmbedder,
    arena: std.heap.ArenaAllocator,

    fn open(dir: std.fs.Dir) !*Reopened {
        const self = try testing_allocator.create(Reopened);
        errdefer testing_allocator.destroy(self);
        self.arena = std.heap.ArenaAllocator.init(testing_allocator);
        errdefer self.arena.deinit();
        const te = try testEmbedder(testing_allocator);
        errdefer testing_allocator.destroy(te.e);
        self.embedder = te.e;
        self.db = try TestVecDB.init(self.arena.allocator(), dir, .{});
        return self;
    }

    fn close(self: *Reopened) void {
        self.db.deinit();
        testing_allocator.destroy(self.embedder);
        self.arena.deinit();
        testing_allocator.destroy(self);
    }
};

/// Like `expectSearchResultsIgnoresimilarity`, but order-independent. Results that tie on
/// similarity come back in an arbitrary order, which these tests don't care about.
fn expectSearchResultsUnordered(expected: []const SearchResult, actual: []const SearchResult) !void {
    if (expected.len != actual.len) {
        std.debug.print(
            "slice lengths differ: expected {d}, found {d}\n",
            .{ expected.len, actual.len },
        );
        return error.TestExpectedEqual;
    }
    var matched: [128]bool = .{false} ** 128;
    assert(actual.len <= matched.len);
    outer: for (expected) |e| {
        for (actual, 0..) |a, i| {
            if (matched[i]) continue;
            if (std.mem.eql(u8, e.path, a.path) and e.start_i == a.start_i and e.end_i == a.end_i) {
                matched[i] = true;
                continue :outer;
            }
        }
        std.debug.print(
            "no result for {{ .path = {s}, .start_i = {d}, .end_i = {d} }}\n",
            .{ e.path, e.start_i, e.end_i },
        );
        return error.TestExpectedEqual;
    }
}

test "embedText persists to disk" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "hello");

    // `db` is deliberately left open: this asserts that embedText itself flushed the vectors,
    // not that some later teardown step did.
    const reopened = try Reopened.open(tmpD.dir);
    defer reopened.close();

    var buf: [10]SearchResult = undefined;
    const found = try reopened.db.search("hello", &buf);
    try expectEqual(1, found);
    try expectSearchResultsUnordered(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 5 },
    }, buf[0..found]);

    try reopened.db.validate();
}

test "embedText persists multiple sentences and paths to disk" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    try db.embedText("test1.md", "pizza. pizza. pizza.");
    try db.embedText("test2.md", "pizza");

    const reopened = try Reopened.open(tmpD.dir);
    defer reopened.close();

    var buf: [10]SearchResult = undefined;
    const found = try reopened.db.search("pizza", &buf);
    try expectEqual(4, found);
    try expectSearchResultsUnordered(&[_]SearchResult{
        .{ .path = "test1.md", .start_i = 0, .end_i = 5 },
        .{ .path = "test1.md", .start_i = 7, .end_i = 12 },
        .{ .path = "test1.md", .start_i = 14, .end_i = 19 },
        .{ .path = "test2.md", .start_i = 0, .end_i = 5 },
    }, buf[0..found]);

    try reopened.db.validate();
}

test "embedText persists removals to disk" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "hello");
    // Replacing the contents drops the old vector; that drop has to reach disk too.
    try db.embedText(path, "flatiron");

    const reopened = try Reopened.open(tmpD.dir);
    defer reopened.close();

    var buf: [10]SearchResult = undefined;
    try expectEqual(0, try reopened.db.search("hello", &buf));
    const found = try reopened.db.search("flatiron", &buf);
    try expectEqual(1, found);
    try expectSearchResultsUnordered(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 8 },
    }, buf[0..found]);

    try reopened.db.validate();
}

test "embedTextAsync persists to disk after shutdown" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedTextAsync(path, "pizza. pizza. pizza.");
    // shutdown drains the queue, so everything queued above must be on disk once it returns.
    db.shutdown();

    const reopened = try Reopened.open(tmpD.dir);
    defer reopened.close();

    var buf: [10]SearchResult = undefined;
    const found = try reopened.db.search("pizza", &buf);
    try expectEqual(3, found);
    try expectSearchResultsUnordered(&[_]SearchResult{
        .{ .path = path, .start_i = 0, .end_i = 5 },
        .{ .path = path, .start_i = 7, .end_i = 12 },
        .{ .path = path, .start_i = 14, .end_i = 19 },
    }, buf[0..found]);

    try reopened.db.validate();
}

test "embedTextAsync persists every queued job after shutdown" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    const N = 10;
    var path_bufs: [N][16]u8 = undefined;
    var expected: [N]SearchResult = undefined;
    for (0..N) |i| {
        const path = try bufPrint(&path_bufs[i], "test{d}.md", .{i});
        try db.embedTextAsync(path, "pizza");
        expected[i] = .{ .path = path, .start_i = 0, .end_i = 5 };
    }
    db.shutdown();

    const reopened = try Reopened.open(tmpD.dir);
    defer reopened.close();

    var buf: [N * 2]SearchResult = undefined;
    const found = try reopened.db.search("pizza", &buf);
    // Localises a shortfall: the store, the index, and the path map each have to hold all ten
    // for the search to, and the counts say which one did not.
    try expectEqual(N, reopened.db.note_id_map.count());
    try expectEqual(N, reopened.db.vec_storage.len());
    try expectEqual(N, reopened.db.codes.len());
    try expectEqual(N, found);
    try expectSearchResultsUnordered(&expected, buf[0..found]);

    try reopened.db.validate();
}

test "empty inputs" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    var res: [1]SearchResult = undefined;

    try expectEqual(0, db.search("", &res));
    try expectEqual(0, db.uniqueSearch("", &res));

    var highlights: [2]usize = undefined;
    try db.populateHighlights("", "", &.{});
    try db.populateHighlights("", "book", &highlights);
    try db.populateHighlights("book", "", &highlights);
    try db.populateHighlights("book", "book", &.{});
    try db.populateHighlights("book", "", &.{});
    try db.populateHighlights("", "book", &.{});
    try db.populateHighlights("", "", &highlights);

    try expectEqual(Error.InvalidPath, db.removePath(""));

    try expectEqual(Error.InvalidPath, db.renamePath("foo", ""));
    try expectEqual(Error.InvalidPath, db.renamePath("", "foo"));

    try expectEqual(Error.InvalidPath, db.embedText("", ""));
    try db.embedText("foo", "");
    try expectEqual(Error.InvalidPath, db.embedText("", "foo"));

    try expectEqual(Error.InvalidPath, db.embedTextAsync("", ""));
    try db.embedTextAsync("foo", "");
    try expectEqual(Error.InvalidPath, db.embedTextAsync("", "foo"));

    // There should be something present at foo
    try db.embedText("foo", "bar");
    try expectEqual(0, db.search("foo", &.{}));
    try expectEqual(0, db.uniqueSearch("foo", &.{}));
}

const std = @import("std");
const testing_allocator = std.testing.allocator;
const expectEqual = std.testing.expectEqual;
const expectEqualSlices = std.testing.expectEqualSlices;
const assert = std.debug.assert;

const config = @import("config");
const tracy = @import("tracy");

const bufPrint = std.fmt.bufPrint;
const embed = @import("embed.zig");
const expect = std.testing.expect;
const EmbeddingModel = embed.EmbeddingModel;
const isAlphanumeric = std.ascii.isAlphanumeric;
const note_id_map_mod = @import("note_id_map.zig");
const NoteID = note_id_map_mod.NoteID;
const NoteIdMap = note_id_map_mod.NoteIdMap;

const NLEmbedder = embed.NLEmbedder;
const MpnetEmbedder = embed.MpnetEmbedder;
/// Takes its operands by pointer. A @Vector(768, f32) is padded to 4096 bytes, so the
/// by-value form copies 8 KB per candidate -- see experiments/results/storebench.md, where
/// that was the whole cost of a whole-store scan.
const storedDotAt = @import("vec_util.zig").storedDotAt;
const quant32to16 = @import("vec_util.zig").quant32to16;
const quant32toi8 = @import("vec_util.zig").quant32toi8;
const spawn = Thread.spawn;
const Thread = std.Thread;
const types = @import("types.zig");
const UniqueCircularBuffer = util.UniqueCircularBuffer;
const util = @import("util.zig");
const VectorID = types.VectorID;
const vstore = @import("vstore.zig");
const codes = @import("codes.zig");

test "the code index stays in step with the store across re-embeds and removals" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();

    // Two documents, re-embedded with different content, then one removed. Every one of those
    // paths has to touch both halves; `validate` is what catches it if one does not.
    try db.embedText("a.md", "pizza. pasta. bread.");
    try db.embedText("b.md", "cars. trains. planes. boats.");
    try db.validate();
    try expectEqual(db.vec_storage.len(), db.codes.len());

    // Re-embedding shorter must free the surplus in both.
    try db.embedText("a.md", "pizza.");
    try db.validate();
    try expectEqual(@as(usize, 5), db.vec_storage.len());
    try expectEqual(@as(usize, 5), db.codes.len());

    // ...and longer must claim reused slots in both.
    try db.embedText("a.md", "pizza. pasta. bread. cheese. olives.");
    try db.validate();
    try expectEqual(@as(usize, 9), db.vec_storage.len());
    try expectEqual(@as(usize, 9), db.codes.len());

    try db.removePath("b.md");
    try db.validate();
    try expectEqual(@as(usize, 5), db.vec_storage.len());
    try expectEqual(@as(usize, 5), db.codes.len());

    // The removed document must be gone from results, not merely from the store.
    var buf: [10]SearchResult = undefined;
    const n = try db.search("trains", &buf);
    for (buf[0..n]) |r| try std.testing.expect(!std.mem.eql(u8, r.path, "b.md"));
}

test "the index is rebuilt on reopen and finds what it found before" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();

    const text = "pizza is delicious. trains are fast. the sky is blue.";
    var before: [4]SearchResult = undefined;
    var n_before: usize = 0;
    {
        var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
        defer db.deinit();
        try db.embedText("a.md", text);
        n_before = try db.search("trains are fast", &before);
        try std.testing.expect(n_before > 0);
        // `path` borrows from this db's note_id_map, which the defer above is about to free.
        for (before[0..n_before]) |*r| r.path = try arena.allocator().dupe(u8, r.path);
    }

    // Nothing persists the codes yet, so this exercises the rebuild path in `init`.
    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();
    try db.validate();
    try expectEqual(db.vec_storage.len(), db.codes.len());

    var after: [4]SearchResult = undefined;
    const n_after = try db.search("trains are fast", &after);
    try expectEqual(n_before, n_after);
    for (before[0..n_before], after[0..n_after]) |a, b| {
        try std.testing.expectEqualStrings(a.path, b.path);
        try expectEqual(a.start_i, b.start_i);
        try expectEqual(a.end_i, b.end_i);
    }
}

test "the code index is saved on close and loaded on reopen" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();

    const text = "pizza is delicious. trains are fast. the sky is blue. cats sleep often.";
    {
        var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
        defer db.deinit();
        try db.embedText("a.md", text);
    }

    // The file exists only because deinit wrote it, and it is named after the database.
    var name_buf: [256]u8 = undefined;
    const name = try codesPath(&name_buf, TestVecDB.embedderPathForTest());
    try tmpD.dir.access(name, .{});

    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();
    try db.validate();
    try expectEqual(db.vec_storage.len(), db.codes.len());

    // Loading consumed the file: an index on disk is only valid while no process holds it.
    try std.testing.expectError(error.FileNotFound, tmpD.dir.access(name, .{}));

    var buf: [4]SearchResult = undefined;
    try std.testing.expect(try db.search("trains are fast", &buf) > 0);
}

test "a stale index is rejected rather than trusted" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();

    var name_buf: [256]u8 = undefined;
    const name = try codesPath(&name_buf, TestVecDB.embedderPathForTest());

    {
        var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
        defer db.deinit();
        try db.embedText("a.md", "pizza is delicious.");
    }
    // An index written for a one-document store.
    try tmpD.dir.access(name, .{});
    const stale = try tmpD.dir.readFileAlloc(arena.allocator(), name, 1 << 24);

    {
        var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
        defer db.deinit();
        try db.embedText("b.md", "trains are fast. the sky is blue.");
    }
    // Put the one-document index back over the two-document one.
    try tmpD.dir.writeFile(.{ .sub_path = name, .data = stale });

    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();
    // Rejected on the stamp, so the index was rebuilt and covers both documents.
    try db.validate();
    try expectEqual(db.vec_storage.len(), db.codes.len());

    var buf: [4]SearchResult = undefined;
    try std.testing.expect(try db.search("trains are fast", &buf) > 0);
    try std.testing.expect(try db.search("pizza is delicious", &buf) > 0);
}

test "a missing index is rebuilt, and answers the same as a loaded one" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();

    var name_buf: [256]u8 = undefined;
    const name = try codesPath(&name_buf, TestVecDB.embedderPathForTest());
    const text = "pizza is delicious. trains are fast. the sky is blue. cats sleep often.";

    {
        var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
        defer db.deinit();
        try db.embedText("a.md", text);
    }

    var loaded_buf: [4]SearchResult = undefined;
    var n_loaded: usize = 0;
    {
        var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
        defer db.deinit();
        n_loaded = try db.search("trains are fast", &loaded_buf);
        for (loaded_buf[0..n_loaded]) |*r| r.path = try arena.allocator().dupe(u8, r.path);
    }

    // That reopen consumed the index, and this deinit wrote a fresh one -- delete it so the
    // next open takes the rebuild path instead.
    tmpD.dir.deleteFile(name) catch {};

    var db = try TestVecDB.init(testing_allocator, tmpD.dir, .{});
    defer db.deinit();
    var rebuilt: [4]SearchResult = undefined;
    const n_rebuilt = try db.search("trains are fast", &rebuilt);

    try expectEqual(n_loaded, n_rebuilt);
    for (loaded_buf[0..n_loaded], rebuilt[0..n_rebuilt]) |a, b| {
        try std.testing.expectEqualStrings(a.path, b.path);
        try expectEqual(a.start_i, b.start_i);
        try expectEqual(a.similarity, b.similarity);
    }
}
