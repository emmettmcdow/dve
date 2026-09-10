const MAX_NOTE_LEN: usize = std.math.maxInt(u32);

pub const Error = error{ NotQueuedShuttingDown, InvalidPath };

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
        pub const VecStorage = vec_storage.Storage(VEC_SZ, STORED_VEC_TYPE);
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
        note_id_map: *NoteIdMap,
        basedir: std.fs.Dir,
        allocator: std.mem.Allocator,
        work_queue: *WorkQueue,
        work_queue_thread: Thread,
        work_queue_mutex: Thread.Mutex = .{},
        work_queue_condition: Thread.Condition = .{},
        work_queue_running: bool,

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

            var vecs = try VecStorage.init(allocator, basedir, .{});
            try vecs.load(embedder.path);
            const wq = try WorkQueue.init(allocator, 1024);

            const note_id_map = try allocator.create(NoteIdMap);
            errdefer allocator.destroy(note_id_map);
            note_id_map.* = try NoteIdMap.init(allocator, basedir);

            const self = try allocator.create(Self);
            self.* = .{
                .base_embedder = base_embedder,
                .embedder = embedder,
                .vec_storage = vecs,
                .note_id_map = note_id_map,
                .basedir = basedir,
                .allocator = allocator,
                .work_queue = wq,
                .work_queue_thread = try spawn(.{}, Self.workQueueRun, .{self}),
                .work_queue_running = true,
            };
            return self;
        }
        pub fn deinit(self: *Self) void {
            if (self.work_queue_running) {
                self.shutdown();
            }
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
            const query_vec = toStored(@field(query_vec_union, @tagName(embedding_model)).*);
            const vec_res = try arena.allocator().alloc(VecStorage.SearchEntry, max_results);

            debugSearchHeader(query);
            const found_n = try self.vec_storage.search(
                query_vec,
                vec_res,
                self.embedder.threshold,
            );
            for (0..found_n) |i| {
                const p = self.note_id_map.getPath(vec_res[i].row.note_id) orelse continue;
                buf[i] = SearchResult{
                    .path = p,
                    .start_i = vec_res[i].row.start_i,
                    .end_i = vec_res[i].row.end_i,
                    .similarity = vec_res[i].similarity,
                };
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
            const query_vec = toStored(@field(query_vec_union, @tagName(embedding_model)).*);

            debugSearchHeader(query);
            var search_results: [1000]VecStorage.SearchEntry = undefined;
            const found_n = try self.vec_storage.search(
                query_vec,
                &search_results,
                self.embedder.threshold,
            );
            var unique_found_n: usize = 0;
            outer: for (0..@min(found_n, buf.len)) |i| {
                const row = search_results[i].row;
                const path = self.note_id_map.getPath(row.note_id) orelse continue;
                for (0..unique_found_n) |j| {
                    if (std.mem.eql(u8, buf[j].path, path)) continue :outer;
                }
                buf[unique_found_n] = SearchResult{
                    .path = path,
                    .start_i = row.start_i,
                    .end_i = row.end_i,
                    .similarity = search_results[i].similarity,
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

                const similar = vec_storage.cosine_similarity(
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
            return self.vec_storage.save(self.embedder.path);
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
            const old_vecs = try self.vec_storage.vecsForNote(allocator, note_id);
            defer allocator.free(old_vecs);

            // TODO(cutover): remove before putting. Either order is correct -- a `put` can only
            // claim a slot that is already free, so it can never land on a live vector -- but
            // this order removes the old rows *after* the new ones are placed, so the new rows
            // cannot reuse the slots the old ones are about to vacate. Steady state is two
            // generations of every document on disk instead of one. Removing first halves both
            // the file and the in-memory codes array that a search has to scan.
            for (embedded_sentences) |sentence| {
                _ = try self.vec_storage.put(note_id, sentence.start_i, sentence.end_i, sentence.vec.*);
            }

            for (old_vecs) |old_v| {
                self.vec_storage.rm(old_v.id) catch |e| switch (e) {
                    vec_storage.Error.MultipleRemove => continue,
                    else => unreachable,
                };
            }
        }

        /// Validate the vector database is in a good state.
        pub fn validate(self: *Self) !void {
            try self.vec_storage.validate();
        }

        /// Delete the entries associated with a given path.
        pub fn removePath(self: *Self, path: []const u8) !void {
            if (path.len == 0) return Error.InvalidPath;
            if (self.note_id_map.getId(path)) |note_id| {
                self.vec_storage.rmByNoteId(note_id);
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

        /// Searches the vector database with an already-embedded query vector. Behaves like
        /// `search`, but runs no embedding operations.
        pub fn rawVectorSearch(self: *Self, raw_vec: RawVector, buf: []SearchResult) !usize {
            const zone = tracy.beginZone(@src(), .{ .name = "vector.zig:rawVectorSearch" });
            defer zone.end();
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            const vec = toStored(raw_vec);

            const max_results = buf.len;
            const vec_res = try arena.allocator().alloc(VecStorage.SearchEntry, max_results);
            const found_n = try self.vec_storage.search(
                vec,
                vec_res,
                self.embedder.threshold,
            );
            for (0..found_n) |i| {
                const p = self.note_id_map.getPath(vec_res[i].row.note_id) orelse continue;
                buf[i] = SearchResult{
                    .path = p,
                    .start_i = vec_res[i].row.start_i,
                    .end_i = vec_res[i].row.end_i,
                    .similarity = vec_res[i].similarity,
                };
            }

            std.log.info("Found {d} results searching with raw vector", .{found_n});
            return found_n;
        }
    };
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
const TestVector = TestVecDB.VecStorage.Vector;
fn getVectorsForPath(db: *TestVecDB, path: []const u8, buf: []TestVector) !usize {
    const note_id = db.note_id_map.getId(path) orelse return 0;
    const vec_rows = try db.vec_storage.vecsForNote(testing_allocator, note_id);
    defer testing_allocator.free(vec_rows);
    for (vec_rows, 0..) |v, i| {
        buf[i] = db.vec_storage.getVec(v.row.vec_id);
    }
    return vec_rows.len;
}

test "embedText hello" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    const text = "/hello/";
    try db.embedText(path, text);

    try db.validate();
}

test "embedText clear previous" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "hello");

    var buf: [1]SearchResult = undefined;
    try expectEqual(1, try db.search("hello", &buf));
    try db.embedText(path, "flatiron");
    try expectEqual(0, try db.search("hello", &buf));

    try db.validate();
}

test "search" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "apple");
    var initial_vecs: [1]TestVector = undefined;
    try expectEqual(1, try getVectorsForPath(db, path, &initial_vecs));

    try db.embedText(path, "apple");
    var updated_vecs: [1]TestVector = undefined;
    try expectEqual(1, try getVectorsForPath(db, path, &updated_vecs));

    try std.testing.expect(@reduce(.And, initial_vecs[0] == updated_vecs[0]));

    try db.validate();
}

test "embedText different input different result" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
    defer db.deinit();

    const path = "test.md";
    try db.embedText(path, "apple");
    var initial_vecs: [1]TestVector = undefined;
    try expectEqual(1, try getVectorsForPath(db, path, &initial_vecs));

    try db.embedText(path, "banana");
    var updated_vecs: [1]TestVector = undefined;
    try expectEqual(1, try getVectorsForPath(db, path, &updated_vecs));

    // Vector should be different (apple != banana)
    try std.testing.expect(!@reduce(.And, initial_vecs[0] == updated_vecs[0]));

    try db.validate();
}

test "embedText updates only changed sentences" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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

    try std.testing.expect(@reduce(.And, initial_vecs[0] == updated_vecs[0]));
    try std.testing.expect(!@reduce(.And, initial_vecs[1] == updated_vecs[1]));
    try std.testing.expect(@reduce(.And, initial_vecs[2] == updated_vecs[2]));

    try db.validate();
}

test "embedText handle multiple remove gracefully" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
    try expectEqual(N, found);
    try expectSearchResultsUnordered(&expected, buf[0..found]);

    try reopened.db.validate();
}

test "empty inputs" {
    var tmpD = std.testing.tmpDir(.{ .iterate = true });
    defer tmpD.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing_allocator);
    defer arena.deinit();
    var db = try TestVecDB.init(arena.allocator(), tmpD.dir, .{});
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
const quant32to16 = @import("vec_util.zig").quant32to16;
const quant32toi8 = @import("vec_util.zig").quant32toi8;
const spawn = Thread.spawn;
const Thread = std.Thread;
const types = @import("types.zig");
const UniqueCircularBuffer = util.UniqueCircularBuffer;
const util = @import("util.zig");
const VectorID = types.VectorID;
const vec_storage = @import("vec_storage.zig");
