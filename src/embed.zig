// Note on Objective-C runtime
// Because we are calling into a garbage-collected language (Objective-C), we need to manage our
// memory accordingly. When a new Object is created (typically through `msgSend(Object...`), the
// runtime needs to know whether it can GC that instance. This is done through
// - Object.retain() - indicates we need this instance, incrementing the refence counter.
// - Object.release() - indicates we are finished with this object, decrementing the ref counter.
// - objc.AutoreleasePool.init() and .deinit() - works like an arena.
// Most allocated objects start with a reference counter of 1, so there is no need to `retain` an
// object most of the time. The only time we need to explicitly retain is when we get an object
// from some parent object which we release. The release cascades down to children.
// AutoreleasePools don't work for explicit allocations.
pub const EmbeddingModel = enum {
    apple_nlembedding,
    mpnet_embedding,
    llama_nomic_embed_text_v1_5_f32,

    pub fn referenceImplementationName(self: EmbeddingModel) []const u8 {
        return switch (self) {
            .mpnet_embedding => MpnetEmbedder.REFERENCE_IMPLEMENTATION_NAME,
            .apple_nlembedding => NLEmbedder.REFERENCE_IMPLEMENTATION_NAME,
            .llama_nomic_embed_text_v1_5_f32 => LlamaNomicEmbedTextV15F32.REFERENCE_IMPLEMENTATION_NAME,
        };
    }

    pub fn vecSize(self: EmbeddingModel) usize {
        return switch (self) {
            .mpnet_embedding => MpnetEmbedder.VEC_SZ,
            .apple_nlembedding => NLEmbedder.VEC_SZ,
            .llama_nomic_embed_text_v1_5_f32 => LlamaNomicEmbedTextV15F32.VEC_SZ,
        };
    }
};

pub const EmbeddingModelOutput = union(EmbeddingModel) {
    apple_nlembedding: *const @Vector(NLEmbedder.VEC_SZ, NLEmbedder.VEC_TYPE),
    mpnet_embedding: *const @Vector(MpnetEmbedder.VEC_SZ, MpnetEmbedder.VEC_TYPE),
    llama_nomic_embed_text_v1_5_f32: *const @Vector(LlamaNomicEmbedTextV15F32.VEC_SZ, LlamaNomicEmbedTextV15F32.VEC_TYPE),

    pub fn slice(self: EmbeddingModelOutput) []const f32 {
        return switch (self) {
            .mpnet_embedding => |v| @as(*const [MpnetEmbedder.VEC_SZ]f32, @ptrCast(v)),
            .apple_nlembedding => |v| @as(*const [NLEmbedder.VEC_SZ]f32, @ptrCast(v)),
            .llama_nomic_embed_text_v1_5_f32 => |v| @as(*const [LlamaNomicEmbedTextV15F32.VEC_SZ]f32, @ptrCast(v)),
        };
    }

    pub fn vecSize(self: EmbeddingModelOutput) usize {
        return switch (self) {
            .mpnet_embedding => MpnetEmbedder.VEC_SZ,
            .apple_nlembedding => NLEmbedder.VEC_SZ,
            .llama_nomic_embed_text_v1_5_f32 => LlamaNomicEmbedTextV15F32.VEC_SZ,
        };
    }
};

pub const Embedder = struct {
    ptr: *anyopaque,
    splitFn: *const fn (ptr: *anyopaque, contents: []const u8) SentenceSpliterator,
    embedFn: EmbedFn,
    embedBatchFn: EmbedBatchFn,
    deinitFn: *const fn (self: *anyopaque) void,

    id: EmbeddingModel,
    threshold: f32,
    strict_threshold: f32,
    path: []const u8,

    pub fn split(self: *Embedder, contents: []const u8) SentenceSpliterator {
        return self.splitFn(self.ptr, contents);
    }

    pub fn embed(
        self: *Embedder,
        allocator: Allocator,
        contents: []const u8,
    ) !?EmbeddingModelOutput {
        return self.embedFn(self.ptr, allocator, contents);
    }

    /// Embeds every string in `strs` at once. The returned slice lines up with
    /// `strs`: entry i is what `embed(allocator, strs[i])` would give, null for
    /// the strings `embed` would skip. The slice and every vector in it come
    /// from `allocator`; as with `embed`, an arena is the expected choice.
    pub fn embedBatch(
        self: *Embedder,
        allocator: Allocator,
        strs: []const []const u8,
    ) ![]?EmbeddingModelOutput {
        return self.embedBatchFn(self.ptr, allocator, strs);
    }

    pub fn deinit(self: *Embedder) void {
        self.deinitFn(self.ptr);
    }
};

pub const EmbedFn = *const fn (
    ptr: *anyopaque,
    allocator: Allocator,
    str: []const u8,
) anyerror!?EmbeddingModelOutput;

pub const EmbedBatchFn = *const fn (
    ptr: *anyopaque,
    allocator: Allocator,
    strs: []const []const u8,
) anyerror![]?EmbeddingModelOutput;

/// A batch in name only: calls `embedOne` once per string. For backends with no
/// native batching, so that every Embedder can answer `embedBatch`.
fn sequentialEmbedBatch(comptime embedOne: EmbedFn) EmbedBatchFn {
    return struct {
        fn embedBatch(
            ptr: *anyopaque,
            allocator: Allocator,
            strs: []const []const u8,
        ) ![]?EmbeddingModelOutput {
            const outs = try allocator.alloc(?EmbeddingModelOutput, strs.len);
            errdefer allocator.free(outs);
            for (strs, outs) |str, *out| out.* = try embedOne(ptr, allocator, str);
            return outs;
        }
    }.embedBatch;
}

//**************************************************************************************** Embedder
/// Compiled models already loaded in this process, keyed by the path they were resolved
/// from.
///
/// `MLModel.compileModelAtURL` writes the compiled model to a *temporary* directory, and
/// Apple's contract is that the caller moves it if it needs to persist -- the system is free
/// to reclaim it. This code never moved it, and compiled afresh on every `init`.
///
/// That is what made the mpnet tests flaky. Repeatedly loading the model in one process
/// aborted inside MetalPerformanceShadersGraph with `shape.count = 0 != strides.count = 3`,
/// a model whose output descriptions came back with no shape at all. Measured: 0 failures in
/// 28 runs loading a precompiled `.mlmodelc` against 6 in 40 runs compiling each time, with
/// the compile step as the only difference. Concurrency was not involved -- a purely
/// sequential reload reproduces it, and 200 concurrent predictions through one model do not.
///
/// Compiling once per process removes the repeated compile entirely, and makes every `init`
/// after the first nearly free, which the test suite feels more than anything else.
///
/// Entries are retained for the life of the process and never released. Tearing CoreML
/// objects down at exit is its own hazard -- see the note in `llama_bridge.c` about ggml's
/// Metal teardown asserting against static destructors -- and one retained model is a far
/// cheaper problem than that.
const ModelCache = struct {
    /// Guards the tables below.
    var mutex: Mutex = .init;
    var by_path: ?std.StringHashMap(Object) = null;
    /// Keys that were asked for and are not there: a batch function in a model converted
    /// without any. Remembered so that every `init` after the first does not ask again.
    var missing: ?std.StringHashMap(void) = null;
    /// Parsed tokenizers, keyed and retained the same way as the models.
    var tokenizers: ?std.StringHashMap(*WordPieceTokenizer) = null;
    /// Held across `MLModel` prediction. Process-wide because the models are.
    var predict_mutex: Mutex = .init;
};

pub const MpnetEmbedder = struct {
    /// The model's default function: one row of `SINGLE.seq` tokens.
    model: Object,
    /// The model's batch functions, loaded by the first `embedBatch` rather than by `init`:
    /// an embedder that only ever embeds queries should not pay to load them. Guarded by
    /// `ModelCache.predict_mutex`.
    batch: BatchModels = .unloaded,
    compute_units: ComputeUnits,
    /// Shared with every other embedder that loaded the same file -- see `ModelCache`.
    tokenizer: *tokenizer_mod.WordPieceTokenizer,
    /// Owns the two paths below, and nothing else now that the tokenizer is shared.
    tokenizer_alloc: std.heap.ArenaAllocator,
    io: std.Io,
    /// The files this embedder actually loaded, after option/bundle/exe-relative resolution.
    /// Owned by `tokenizer_alloc`.
    loaded_model_path: [:0]const u8,
    loaded_tokenizer_path: [:0]const u8,

    pub const VEC_SZ = 768;
    pub const VEC_TYPE = f32;
    pub const ID = EmbeddingModel.mpnet_embedding;
    pub const REFERENCE_IMPLEMENTATION_NAME = "sentence-transformers/all-mpnet-base-v2";
    pub const THRESHOLD = 0.36;
    pub const STRICT_THRESHOLD = THRESHOLD + 0.1;
    // Suffixed by storage type: a quantized database is not loadable by an unquantized
    // build, so the two live side by side rather than one refusing the other's file.
    pub const PATH = @tagName(ID) ++ db_suffix ++ ".db";
    pub const MODEL_PATH = "share/all_mpnet_base_v2.mlpackage";
    pub const TOKENIZER_PATH = "share/tokenizer.json";
    pub const BUNDLE_MODEL_PATH = "all_mpnet_base_v2.mlmodelc";
    pub const BUNDLE_TOKENIZER_PATH = "tokenizer.json";
    /// Fetches the model and tokenizer into ./all_mpnet_base_v2/. The release is the one
    /// build.zig.zon pins as `coreml_models`; keep the two, and USAGE.md, in step.
    pub const DOWNLOAD_CMD = "curl -L " ++
        zon.dependencies.coreml_models.url ++
        " | tar -xz all_mpnet_base_v2";
    const MAX_SEQ_LEN = 512;

    /// An input shape the model was converted for. The Neural Engine takes fixed shapes
    /// only, so each one is a separate function in the model package.
    const Shape = struct {
        rows: usize,
        seq: usize,
        /// Function name in the package. Null is the default function.
        function: ?[:0]const u8 = null,
        /// Fewest rows worth a prediction of this shape. A prediction costs the same however
        /// many of its rows are padding, so below this it is cheaper to send the stragglers
        /// through `SINGLE` one at a time. Derived from the measured cost of each shape:
        /// 4.3ms for 1x128, 13.3ms for 32x16 and 28.0ms for 32x32.
        min_rows: usize = 1,
    };

    const SINGLE: Shape = .{ .rows = 1, .seq = 128 };

    /// Ascending by `seq`: a sentence goes to the first shape it fits in, and to `SINGLE`
    /// if it fits in none. Keep in step with `--batch-shapes` in models/gen-coreml.py.
    ///
    /// Short rows are the point. The Neural Engine gets through about 37,000 token slots a
    /// second however they are arranged, and a sentence averages a dozen tokens, so rows of
    /// 128 are nine-tenths padding: measured, 32x128 embeds 255 sentences a second against
    /// 232 for 1x128, where 32x16 embeds 2,400.
    const BATCH_SHAPES = [_]Shape{
        .{ .rows = 32, .seq = 16, .function = "b32_s16", .min_rows = 4 },
        .{ .rows = 32, .seq = 32, .function = "b32_s32", .min_rows = 7 },
    };
    const MAX_ROWS = 32;

    const BatchModels = union(enum) {
        unloaded,
        /// The model has no batch functions, as every `coreml_models` release up to v5.
        /// `embedBatch` then embeds one string at a time.
        unavailable,
        ready: [BATCH_SHAPES.len]Object,
    };

    // MLMultiArrayDataType, whose values are a CoreML API constant: 0x10000 | bit width.
    const MLMultiArrayDataTypeFloat16: i64 = 0x10000 | 16;
    const MLMultiArrayDataTypeFloat32: i64 = 0x10000 | 32;

    pub const InitOptions = struct {
        /// Path to the `.mlpackage` or `.mlmodelc`. Null looks for `BUNDLE_MODEL_PATH` in
        /// the app bundle's resources, then for `MODEL_PATH` beside the executable.
        model_path: ?[]const u8 = null,
        /// Path to `tokenizer.json`. Null is resolved the same way as `model_path`.
        tokenizer_path: ?[]const u8 = null,
        /// Which engines CoreML may schedule the model on. `all` lets CoreML pick, which on
        /// Apple silicon means the Neural Engine for a model that can use it.
        ///
        /// **This default assumes an fp16 model**, which is what `coreml_models` ships from
        /// v5 on. Measured, same sentences:
        ///
        ///     fp16, all              248.8 chunks/sec   0 aborts in 40
        ///     fp32, gpu               51.1              9 aborts in 124
        ///     fp32, cpu+ane           30.7              0          (really the CPU)
        ///
        /// The Neural Engine is fp16-only, so an fp32 model cannot reach it and silently runs
        /// on the CPU at an eighth the speed -- and an fp32 model offered the GPU aborts the
        /// process, MetalPerformanceShadersGraph failing `shape.count = 0 != strides.count =
        /// 3` in roughly one process in twelve. Pointing this at an fp32 model therefore
        /// wants `.cpu_only` or `.cpu_and_neural_engine`, and the slowdown is the price.
        ///
        /// `models/gen-coreml.py` converts fp16 by default and handles the one constant that
        /// makes a naive fp16 conversion of a BERT silently wrong.
        compute_units: ComputeUnits = .all,
    };

    /// MLComputeUnits, whose values land in a CoreML API and so are fixed.
    pub const ComputeUnits = enum(i64) {
        cpu_only = 0,
        cpu_and_gpu = 1,
        all = 2,
        cpu_and_neural_engine = 3,
    };

    // The calling Swift thread wraps this in an AutoreleasePool, so we do not need to release
    // anything here. We only need to retain the model and the rest will be cleaned up.
    pub fn init(io: std.Io, opts: InitOptions) !MpnetEmbedder {
        const init_zone = tracy.beginZone(@src(), .{ .name = "embed.zig:MpnetEmbedder.init" });
        defer init_zone.end();
        const pool = objc.AutoreleasePool.init();
        defer pool.deinit();
        var tokenizer_alloc = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        errdefer tokenizer_alloc.deinit();

        const tokenizer_path: [:0]const u8 = if (opts.tokenizer_path) |p|
            tokenizer_alloc.allocator().dupeSentinel(u8, p, 0) catch return error.TokenizerLoadFailed
        else
            getModelPath(tokenizer_alloc.allocator(), io, TOKENIZER_PATH, BUNDLE_TOKENIZER_PATH) catch {
                return error.TokenizerLoadFailed;
            };

        const tok = try acquireTokenizer(io, tokenizer_path);

        const NSString = objc.getClass("NSString") orelse {
            std.log.err("Failed to get NSString class", .{});
            return error.ObjCClassNotFound;
        };
        const NSURL = objc.getClass("NSURL") orelse {
            std.log.err("Failed to get NSURL class", .{});
            return error.ObjCClassNotFound;
        };
        const fromUTF8 = objc.Sel.registerName("stringWithUTF8String:");
        const fileURLWithPath = objc.Sel.registerName("fileURLWithPath:");

        const full_path: [:0]const u8 = if (opts.model_path) |p|
            tokenizer_alloc.allocator().dupeSentinel(u8, p, 0) catch return error.PathAllocFailed
        else
            preferPrecompiled(
                tokenizer_alloc.allocator(),
                io,
                getModelPath(tokenizer_alloc.allocator(), io, MODEL_PATH, BUNDLE_MODEL_PATH) catch {
                    return error.PathAllocFailed;
                },
            ) catch return error.PathAllocFailed;
        std.Io.Dir.cwd().access(io, full_path, .{}) catch {
            logModelNotFound("mpnet model", full_path, DOWNLOAD_CMD, "model_path");
            return error.ModelNotFound;
        };

        const path_ns = NSString.msgSend(Object, fromUTF8, .{full_path.ptr});
        if (path_ns.value == 0) {
            std.log.err("Failed to create NSString from path", .{});
            return error.NSStringCreateFailed;
        }

        const model_url = NSURL.msgSend(Object, fileURLWithPath, .{path_ns});
        if (model_url.value == 0) {
            std.log.err("Failed to create NSURL from path", .{});
            return error.NSURLCreateFailed;
        }

        const resolved = std.mem.sliceTo(full_path, 0);
        const is_precompiled = std.mem.endsWith(u8, resolved, ".mlmodelc");

        const model = try acquireModel(
            io,
            resolved,
            model_url,
            is_precompiled,
            opts.compute_units,
            null,
        );
        errdefer model.release();
        // CoreML will not tell us which engine it actually chose, so the next best thing is
        // to say which ones it was allowed to choose from. Anything other than `all` means
        // somebody deliberately narrowed it, and `cpu_only` means degraded by construction.
        std.log.info("mpnet: compute units {t}, model {s}", .{ opts.compute_units, resolved });

        return .{
            .model = model,
            .compute_units = opts.compute_units,
            .tokenizer = tok,
            .tokenizer_alloc = tokenizer_alloc,
            .io = io,
            .loaded_model_path = full_path,
            .loaded_tokenizer_path = tokenizer_path,
        };
    }

    /// Returns the tokenizer parsed from `path`. Reads and parses it on the first call for
    /// that path and never again: the vocabulary is read-only once built, so every embedder
    /// in the process can share one, and parsing it was most of what a second `init` cost.
    fn acquireTokenizer(io: std.Io, path: []const u8) !*WordPieceTokenizer {
        ModelCache.mutex.lockUncancelable(io);
        defer ModelCache.mutex.unlock(io);

        if (ModelCache.tokenizers == null) {
            ModelCache.tokenizers =
                std.StringHashMap(*WordPieceTokenizer).init(std.heap.page_allocator);
        }
        if (ModelCache.tokenizers.?.get(path)) |cached| return cached;

        // Lives as long as the cache entry does, which is the life of the process.
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        const json = std.Io.Dir.cwd().readFileAlloc(
            io,
            path,
            allocator,
            .limited(10 * 1024 * 1024),
        ) catch |err| {
            if (err == error.FileNotFound) {
                logModelNotFound("mpnet tokenizer", path, DOWNLOAD_CMD, "tokenizer_path");
                return error.ModelNotFound;
            }
            std.log.err("Failed to read tokenizer.json: {}", .{err});
            return error.TokenizerLoadFailed;
        };

        const tok = allocator.create(WordPieceTokenizer) catch return error.TokenizerLoadFailed;
        tok.init(allocator, json) catch |err| {
            std.log.err("Failed to parse tokenizer.json: {}", .{err});
            return error.TokenizerParseFailed;
        };

        const key = allocator.dupe(u8, path) catch return error.TokenizerLoadFailed;
        ModelCache.tokenizers.?.put(key, tok) catch return error.TokenizerLoadFailed;
        return tok;
    }

    /// Returns the model for `path`, with one retain transferred to the caller. Compiles and
    /// loads it on the first call for that path and never again -- see `ModelCache`.
    ///
    /// `function` picks one function of a multifunction model, null being the default one.
    /// A function the model does not have is `error.ModelFunctionNotFound`, and is expected:
    /// it is how a model converted without batch shapes says so.
    fn acquireModel(
        io: std.Io,
        path: []const u8,
        model_url: Object,
        is_precompiled: bool,
        compute_units: ComputeUnits,
        function: ?[:0]const u8,
    ) !Object {
        ModelCache.mutex.lockUncancelable(io);
        defer ModelCache.mutex.unlock(io);

        if (ModelCache.by_path == null) {
            ModelCache.by_path = std.StringHashMap(Object).init(std.heap.page_allocator);
            ModelCache.missing = std.StringHashMap(void).init(std.heap.page_allocator);
        }
        // Keyed by path and function: the first `compute_units` asked for wins for the life
        // of the process. Nothing here loads one model two ways, and adding it to the key
        // would invite doing it.
        var key_buf: [std.Io.Dir.max_path_bytes + 64]u8 = undefined;
        const lookup_key = if (function) |f|
            std.fmt.bufPrint(&key_buf, "{s}#{s}", .{ path, f }) catch return error.PathAllocFailed
        else
            path;
        if (ModelCache.by_path.?.get(lookup_key)) |cached| return cached.retain();
        if (ModelCache.missing.?.contains(lookup_key)) return error.ModelFunctionNotFound;

        const MLModel = objc.getClass("MLModel") orelse return error.ObjCClassNotFound;
        const compileModelAtURL = objc.Sel.registerName("compileModelAtURL:error:");
        const modelWithContentsOfURL =
            objc.Sel.registerName("modelWithContentsOfURL:configuration:error:");

        const load_url = if (is_precompiled) model_url else compiled: {
            var compile_error: ?*anyopaque = null;
            const compiled_url = MLModel.msgSend(Object, compileModelAtURL, .{
                model_url,
                &compile_error,
            });
            if (compile_error) |err_ptr| {
                const err = Object{ .value = @intFromPtr(err_ptr) };
                const desc_sel = objc.Sel.registerName("localizedDescription");
                const desc = err.msgSend([*:0]const u8, desc_sel, .{});
                std.log.err("Failed to compile CoreML model: {s}", .{desc});
                return error.ModelCompileFailed;
            }
            if (compiled_url.value == 0) {
                std.log.err("Compiled URL is null", .{});
                return error.ModelCompileFailed;
            }
            break :compiled compiled_url;
        };

        const MLModelConfiguration =
            objc.getClass("MLModelConfiguration") orelse return error.ObjCClassNotFound;
        const config = MLModelConfiguration.msgSend(Object, objc.Sel.registerName("alloc"), .{})
            .msgSend(Object, objc.Sel.registerName("init"), .{});
        defer config.release();
        config.msgSend(void, objc.Sel.registerName("setComputeUnits:"), .{
            @backingInt(compute_units),
        });
        if (function) |f| {
            // macOS 15. An older system cannot load a multifunction model's other functions
            // at all, which reads the same from here as the model not having any.
            const setFunctionName = objc.Sel.registerName("setFunctionName:");
            if (!config.msgSend(bool, objc.Sel.registerName("respondsToSelector:"), .{
                setFunctionName,
            })) return error.ModelFunctionNotFound;
            const NSString = objc.getClass("NSString") orelse return error.ObjCClassNotFound;
            config.msgSend(void, setFunctionName, .{
                NSString.msgSend(Object, objc.Sel.registerName("stringWithUTF8String:"), .{f.ptr}),
            });
        }

        var load_error: ?*anyopaque = null;
        const model = MLModel.msgSend(Object, modelWithContentsOfURL, .{
            load_url,
            config,
            &load_error,
        });
        if (function != null and (load_error != null or model.value == 0)) {
            // The table keeps the key, so it is freed only if the table did not take it:
            // an errdefer would free it on the way out through the error below.
            const key = try std.heap.page_allocator.dupe(u8, lookup_key);
            ModelCache.missing.?.put(key, {}) catch |err| {
                std.heap.page_allocator.free(key);
                return err;
            };
            return error.ModelFunctionNotFound;
        }
        if (load_error) |err_ptr| {
            const err = Object{ .value = @intFromPtr(err_ptr) };
            const desc_sel = objc.Sel.registerName("localizedDescription");
            const desc = err.msgSend([*:0]const u8, desc_sel, .{});
            std.log.err("Failed to load CoreML model: {s}", .{desc});
            return error.ModelLoadFailed;
        }
        if (model.value == 0) {
            std.log.err("Model is null", .{});
            return error.ModelLoadFailed;
        }

        // One retain for the cache and one for the caller; the autoreleased original belongs
        // to whatever pool is current.
        const key = try std.heap.page_allocator.dupe(u8, lookup_key);
        errdefer std.heap.page_allocator.free(key);
        try ModelCache.by_path.?.put(key, model.retain());
        return model.retain();
    }

    pub fn init_self(self: *MpnetEmbedder, io: std.Io, opts: InitOptions) !void {
        const obj = try MpnetEmbedder.init(io, opts);
        self.io = obj.io;
        self.model = obj.model;
        self.batch = obj.batch;
        self.compute_units = obj.compute_units;
        self.tokenizer = obj.tokenizer;
        self.tokenizer_alloc = obj.tokenizer_alloc;
        self.loaded_model_path = obj.loaded_model_path;
        self.loaded_tokenizer_path = obj.loaded_tokenizer_path;
    }

    pub fn embedder(self: *MpnetEmbedder) Embedder {
        return .{
            .ptr = self,
            .splitFn = split,
            .embedFn = embed,
            .embedBatchFn = embedBatch,
            .deinitFn = deinitFn,
            .id = ID,
            .threshold = THRESHOLD,
            .strict_threshold = STRICT_THRESHOLD,
            .path = PATH,
        };
    }

    pub fn deinit(self: *MpnetEmbedder) void {
        self.model.release();
        switch (self.batch) {
            .ready => |models| for (models) |m| m.release(),
            .unloaded, .unavailable => {},
        }
        self.tokenizer_alloc.deinit();
    }

    fn deinitFn(ptr: *anyopaque) void {
        const self: *MpnetEmbedder = @ptrCast(@alignCast(ptr));
        self.deinit();
    }

    fn split(self_ptr: *anyopaque, note: []const u8) SentenceSpliterator {
        _ = self_ptr;
        return SentenceSpliterator.init(note);
    }

    fn embed(
        ptr: *anyopaque,
        allocator: Allocator,
        str: []const u8,
    ) !?EmbeddingModelOutput {
        const self: *MpnetEmbedder = @ptrCast(@alignCast(ptr));
        const zone = tracy.beginZone(@src(), .{ .name = "embed.zig:MpnetEmbedder.embed" });
        defer zone.end();

        if (str.len == 0) {
            std.log.info("Skipping embed of zero-length string", .{});
            return null;
        }
        if (!isAlphanumeric(str[0]) or !isAlphanumeric(str[str.len - 1])) {
            std.log.warn("Embedding str with punctuation is likely unexpected -> '{s}'", .{str});
        }

        const token_ids = try self.tokenizer.tokenize(allocator, str);
        defer allocator.free(token_ids);

        const out = try allocVec(allocator);
        errdefer allocator.destroy(out);

        // Serializes prediction, as NLEmbedder has always done. The engine embeds documents
        // on its worker thread while a search embeds the query on the caller's, through one
        // shared Embedder, so concurrent prediction is the normal case rather than an edge
        // one. The lock lives with the model and not the instance because `ModelCache` hands
        // every MpnetEmbedder in a process the same MLModel -- a per-instance lock would
        // guard nothing.
        ModelCache.predict_mutex.lockUncancelable(self.io);
        defer ModelCache.predict_mutex.unlock(self.io);

        try predictRows(
            self.model,
            SINGLE,
            &.{token_ids[0..@min(token_ids.len, SINGLE.seq)]},
            &.{out},
        );
        return EmbeddingModelOutput{ .mpnet_embedding = out };
    }

    /// Embeds the strings several to a prediction, through the model's batch functions.
    ///
    /// Each string goes to the shortest shape its tokens fit in, and those that fit in none
    /// go through `SINGLE` as `embed` would send them. So does the tail of a shape's queue
    /// when it is too short to be worth a mostly-empty prediction -- see `Shape.min_rows`.
    ///
    /// A model with no batch functions gets every string through `SINGLE`, which is what
    /// this was before there were any.
    fn embedBatch(
        ptr: *anyopaque,
        allocator: Allocator,
        strs: []const []const u8,
    ) ![]?EmbeddingModelOutput {
        const self: *MpnetEmbedder = @ptrCast(@alignCast(ptr));
        const zone = tracy.beginZone(@src(), .{ .name = "embed.zig:MpnetEmbedder.embedBatch" });
        defer zone.end();

        const outs = try allocator.alloc(?EmbeddingModelOutput, strs.len);
        errdefer allocator.free(outs);
        @memset(outs, null);
        if (strs.len == 0) return outs;

        // Parallel arrays over the strings that survive filtering: the tokens, the vector
        // to fill, and whether a prediction has filled it yet.
        const Vec = @Vector(VEC_SZ, VEC_TYPE);
        const tokens = try allocator.alloc([]const u32, strs.len);
        const vecs = try allocator.alloc(*Vec, strs.len);
        const done = try allocator.alloc(bool, strs.len);
        var n: usize = 0;
        for (strs, 0..) |str, i| {
            if (str.len == 0) {
                std.log.info("Skipping embed of zero-length string", .{});
                continue;
            }
            if (!isAlphanumeric(str[0]) or !isAlphanumeric(str[str.len - 1])) {
                std.log.warn("Embedding str with punctuation is likely unexpected -> '{s}'", .{str});
            }
            const token_ids = try self.tokenizer.tokenize(allocator, str);
            tokens[n] = token_ids[0..@min(token_ids.len, SINGLE.seq)];
            vecs[n] = try allocVec(allocator);
            done[n] = false;
            outs[i] = EmbeddingModelOutput{ .mpnet_embedding = vecs[n] };
            n += 1;
        }

        var shape_seq_min: usize = 0;
        for (BATCH_SHAPES, 0..) |shape, shape_i| {
            defer shape_seq_min = shape.seq;

            var rows: [MAX_ROWS][]const u32 = undefined;
            var row_vecs: [MAX_ROWS]*Vec = undefined;
            var row_items: [MAX_ROWS]usize = undefined;
            var row_n: usize = 0;
            for (0..n + 1) |item| {
                if (item < n) {
                    if (tokens[item].len <= shape_seq_min or tokens[item].len > shape.seq) continue;
                    rows[row_n] = tokens[item];
                    row_vecs[row_n] = vecs[item];
                    row_items[row_n] = item;
                    row_n += 1;
                    if (row_n < shape.rows) continue;
                } else if (row_n < shape.min_rows) break;

                // The lock is taken per prediction and not across the batch, so that a
                // search embedding its query waits for one prediction, not for a document.
                ModelCache.predict_mutex.lockUncancelable(self.io);
                defer ModelCache.predict_mutex.unlock(self.io);
                const model = (try self.batchModels() orelse break)[shape_i];
                try predictRows(model, shape, rows[0..row_n], row_vecs[0..row_n]);
                for (row_items[0..row_n]) |filled| done[filled] = true;
                row_n = 0;
            }
        }

        for (tokens[0..n], vecs[0..n], done[0..n]) |row, vec, filled| {
            if (filled) continue;
            ModelCache.predict_mutex.lockUncancelable(self.io);
            defer ModelCache.predict_mutex.unlock(self.io);
            try predictRows(self.model, SINGLE, &.{row}, &.{vec});
        }
        return outs;
    }

    /// One vector's worth of memory, aligned for `@Vector(VEC_SZ, VEC_TYPE)`.
    fn allocVec(allocator: Allocator) !*@Vector(VEC_SZ, VEC_TYPE) {
        const Vec = @Vector(VEC_SZ, VEC_TYPE);
        const buf = try allocator.alignedAlloc(VEC_TYPE, std.mem.Alignment.of(Vec), VEC_SZ);
        return @ptrCast(buf.ptr);
    }

    /// The batch functions, loading them on the first call. Null when the model has none.
    /// The caller holds `ModelCache.predict_mutex`.
    fn batchModels(self: *MpnetEmbedder) !?[BATCH_SHAPES.len]Object {
        switch (self.batch) {
            .ready => |models| return models,
            .unavailable => return null,
            .unloaded => {},
        }
        const pool = objc.AutoreleasePool.init();
        defer pool.deinit();

        const NSString = objc.getClass("NSString") orelse return error.ObjCClassNotFound;
        const NSURL = objc.getClass("NSURL") orelse return error.ObjCClassNotFound;
        const model_url = NSURL.msgSend(Object, objc.Sel.registerName("fileURLWithPath:"), .{
            NSString.msgSend(Object, objc.Sel.registerName("stringWithUTF8String:"), .{
                self.loaded_model_path.ptr,
            }),
        });
        const path = std.mem.sliceTo(self.loaded_model_path, 0);

        var models: [BATCH_SHAPES.len]Object = undefined;
        for (BATCH_SHAPES, 0..) |shape, i| {
            models[i] = acquireModel(
                self.io,
                path,
                model_url,
                std.mem.endsWith(u8, path, ".mlmodelc"),
                self.compute_units,
                shape.function,
            ) catch |err| {
                for (models[0..i]) |m| m.release();
                if (err != error.ModelFunctionNotFound) return err;
                std.log.info(
                    "mpnet: model has no '{s}' function, embedding one string at a time",
                    .{shape.function.?},
                );
                self.batch = .unavailable;
                return null;
            };
        }
        self.batch = .{ .ready = models };
        return models;
    }

    /// Runs one prediction of `shape` and writes the pooled, normalized vector for
    /// `rows[i]` to `outs[i]`. `rows` may be shorter than the shape; the rest is padding.
    /// Every row must be non-empty and fit in `shape.seq`.
    ///
    /// The caller holds `ModelCache.predict_mutex`.
    fn predictRows(
        model: Object,
        shape: Shape,
        rows: []const []const u32,
        outs: []const *@Vector(VEC_SZ, VEC_TYPE),
    ) !void {
        assert(rows.len == outs.len);
        assert(rows.len > 0 and rows.len <= shape.rows);
        const pool = objc.AutoreleasePool.init();
        defer pool.deinit();

        const MLMultiArray = objc.getClass("MLMultiArray") orelse return error.ObjCClassNotFound;
        const NSNumber = objc.getClass("NSNumber") orelse return error.ObjCClassNotFound;
        const NSArray = objc.getClass("NSArray") orelse return error.ObjCClassNotFound;
        const MLDictionaryFeatureProvider = objc.getClass("MLDictionaryFeatureProvider") orelse return error.ObjCClassNotFound;
        const NSDictionary = objc.getClass("NSDictionary") orelse return error.ObjCClassNotFound;
        const MLFeatureValue = objc.getClass("MLFeatureValue") orelse return error.ObjCClassNotFound;
        const NSString = objc.getClass("NSString") orelse return error.ObjCClassNotFound;

        const numberWithInt = objc.Sel.registerName("numberWithInt:");
        const arrayWithObjects = objc.Sel.registerName("arrayWithObjects:count:");
        const initWithShape = objc.Sel.registerName("initWithShape:dataType:error:");
        const alloc_sel = objc.Sel.registerName("alloc");
        const initWithDictionary = objc.Sel.registerName("initWithDictionary:error:");
        const predictionFromFeatures = objc.Sel.registerName("predictionFromFeatures:error:");
        const featureValueForName = objc.Sel.registerName("featureValueForName:");
        const multiArrayValue_sel = objc.Sel.registerName("multiArrayValue");
        const dataPointer_sel = objc.Sel.registerName("dataPointer");
        const dataType_sel = objc.Sel.registerName("dataType");
        const featureValueWithMultiArray = objc.Sel.registerName("featureValueWithMultiArray:");
        const fromUTF8 = objc.Sel.registerName("stringWithUTF8String:");
        const dictionaryWithObjects = objc.Sel.registerName("dictionaryWithObjects:forKeys:count:");

        const rows_num = NSNumber.msgSend(Object, numberWithInt, .{@as(i32, @intCast(shape.rows))});
        const seq_num = NSNumber.msgSend(Object, numberWithInt, .{@as(i32, @intCast(shape.seq))});
        var shape_arr = [_]Object{ rows_num, seq_num };
        const shape_ns = NSArray.msgSend(
            Object,
            arrayWithObjects,
            .{ @as([*]Object, &shape_arr), @as(usize, 2) },
        );

        const MLMultiArrayDataTypeInt32: i64 = 0x20000 | 32;

        var input_err: ?*anyopaque = null;
        const input_ids_array = MLMultiArray.msgSend(Object, alloc_sel, .{}).msgSend(
            Object,
            initWithShape,
            .{ shape_ns, MLMultiArrayDataTypeInt32, &input_err },
        );
        defer input_ids_array.release();
        if (input_err != null) return error.MLMultiArrayInitFailed;
        if (input_ids_array.value == 0) return error.MLMultiArrayInitFailed;

        const attention_mask_array = MLMultiArray.msgSend(Object, alloc_sel, .{}).msgSend(
            Object,
            initWithShape,
            .{ shape_ns, MLMultiArrayDataTypeInt32, &input_err },
        );
        defer attention_mask_array.release();
        if (input_err != null) return error.MLMultiArrayInitFailed;
        if (attention_mask_array.value == 0) return error.MLMultiArrayInitFailed;

        // Row-major, [rows, seq]. Padding is a zero id under a zero mask, both within a
        // row and for the rows past `rows.len`.
        const slot_n = shape.rows * shape.seq;
        const ids = input_ids_array.msgSend([*]i32, dataPointer_sel, .{})[0..slot_n];
        const mask = attention_mask_array.msgSend([*]i32, dataPointer_sel, .{})[0..slot_n];
        @memset(ids, 0);
        @memset(mask, 0);
        for (rows, 0..) |row, r| {
            assert(row.len > 0 and row.len <= shape.seq);
            for (row, 0..) |token, t| {
                ids[r * shape.seq + t] = @intCast(token);
                mask[r * shape.seq + t] = 1;
            }
        }

        const input_ids_fv = MLFeatureValue.msgSend(
            Object,
            featureValueWithMultiArray,
            .{input_ids_array},
        );
        const attention_mask_fv = MLFeatureValue.msgSend(
            Object,
            featureValueWithMultiArray,
            .{attention_mask_array},
        );

        const input_ids_key = NSString.msgSend(Object, fromUTF8, .{"input_ids"});
        const attention_mask_key = NSString.msgSend(Object, fromUTF8, .{"attention_mask"});

        var keys = [_]Object{ input_ids_key, attention_mask_key };
        var values = [_]Object{ input_ids_fv, attention_mask_fv };

        const features_dict = NSDictionary.msgSend(Object, dictionaryWithObjects, .{
            @as([*]Object, &values),
            @as([*]Object, &keys),
            @as(usize, 2),
        });

        var provider_err: ?*anyopaque = null;
        const feature_provider = MLDictionaryFeatureProvider.msgSend(
            Object,
            alloc_sel,
            .{},
        ).msgSend(
            Object,
            initWithDictionary,
            .{ features_dict, &provider_err },
        );
        defer feature_provider.release();
        if (provider_err != null) return error.FeatureProviderInitFailed;
        if (feature_provider.value == 0) return error.FeatureProviderInitFailed;

        // Note to future me: This prediction section is by far the slowest section. Should you
        // choose to optimize it, look here first.
        var pred_err: ?*anyopaque = null;
        const prediction = model.msgSend(
            Object,
            predictionFromFeatures,
            .{ feature_provider, &pred_err },
        );
        if (pred_err) |err_ptr| {
            const err_obj = Object{ .value = @intFromPtr(err_ptr) };
            const desc_sel = objc.Sel.registerName("localizedDescription");
            const desc_ns = err_obj.msgSend(Object, desc_sel, .{});
            const utf8_sel = objc.Sel.registerName("UTF8String");
            const desc = desc_ns.msgSend([*:0]const u8, utf8_sel, .{});
            std.log.err("Prediction failed: {s}", .{desc});
            return error.PredictionFailed;
        }
        if (prediction.value == 0) {
            std.log.err("Prediction returned null (no error reported)", .{});
            return error.PredictionFailed;
        }
        // End of the prediction block

        const output_key = NSString.msgSend(Object, fromUTF8, .{"last_hidden_state"});
        const output_fv = prediction.msgSend(Object, featureValueForName, .{output_key});
        if (output_fv.value == 0) {
            std.log.err("Output feature value is null", .{});
            return error.OutputNotFound;
        }

        const output_array = output_fv.msgSend(Object, multiArrayValue_sel, .{});
        if (output_array.value == 0) {
            std.log.err("Output multi array is null", .{});
            return error.OutputNotFound;
        }

        // Output shape is [rows, seq, VEC_SZ], read below as row-major with no gaps. CoreML
        // does not promise that -- an array may carry strides wider than its shape -- so it
        // is checked rather than assumed: reading a padded layout as a packed one would
        // hand back vectors assembled from the wrong tokens, and nothing would fail.
        const strides = output_array.msgSend(Object, objc.Sel.registerName("strides"), .{});
        const expected_strides = [_]i64{ @intCast(shape.seq * VEC_SZ), VEC_SZ, 1 };
        for (expected_strides, 0..) |expected, axis| {
            const stride = strides
                .msgSend(Object, objc.Sel.registerName("objectAtIndex:"), .{axis})
                .msgSend(i64, objc.Sel.registerName("integerValue"), .{});
            if (stride != expected) {
                std.log.err(
                    "Unexpected CoreML output stride {d} on axis {d}, wanted {d}\n",
                    .{ stride, axis, expected },
                );
                return error.UnsupportedOutputLayout;
            }
        }

        // The element type has to be asked for, not assumed. A model converted with
        // `compute_precision=FLOAT16` -- which is what the Neural Engine requires, and what
        // makes this model 4.4x faster -- hands back a Float16 array, and reading those bytes
        // as f32 does not fail, it silently returns nonsense: measured 0.13 cosine against
        // the same sentence through the fp32 model. Assuming f32 is what tied this embedder
        // to one particular conversion.
        const data_type = output_array.msgSend(i64, dataType_sel, .{});

        for (rows, outs, 0..) |row, out, r| {
            const row_start = r * shape.seq * VEC_SZ;

            // Mean pooling: average only over real (non-padding) token positions.
            const zero_vec: @Vector(VEC_SZ, VEC_TYPE) = @splat(0.0);
            var sum_vec = zero_vec;
            switch (data_type) {
                MLMultiArrayDataTypeFloat32 => {
                    const data_ptr = output_array.msgSend([*]const f32, dataPointer_sel, .{});
                    for (0..row.len) |t| {
                        const token: @Vector(VEC_SZ, VEC_TYPE) =
                            data_ptr[row_start + t * VEC_SZ ..][0..VEC_SZ].*;
                        sum_vec += token;
                    }
                },
                MLMultiArrayDataTypeFloat16 => {
                    const data_ptr = output_array.msgSend([*]const f16, dataPointer_sel, .{});
                    for (0..row.len) |t| {
                        const half: @Vector(VEC_SZ, f16) =
                            data_ptr[row_start + t * VEC_SZ ..][0..VEC_SZ].*;
                        sum_vec += @floatCast(half);
                    }
                },
                else => {
                    std.log.err("Unsupported CoreML output dataType {d}", .{data_type});
                    return error.UnsupportedOutputDataType;
                },
            }
            const count: @Vector(VEC_SZ, VEC_TYPE) = @splat(@floatFromInt(row.len));
            const mean_vec = sum_vec / count;

            // L2 normalize
            const dot = @reduce(.Add, mean_vec * mean_vec);
            const norm: @Vector(VEC_SZ, VEC_TYPE) = @splat(@sqrt(dot));
            out.* = mean_vec / norm;
        }
    }
};

/// Says where a model file was expected and how to get one. Every embedder that loads files
/// reports a missing one through here, so the advice reads the same whichever model it is.
fn logModelNotFound(
    comptime what: []const u8,
    path: []const u8,
    comptime download_cmd: []const u8,
    comptime option: []const u8,
) void {
    std.log.err(
        \\{s} not found at '{s}'.
        \\Download it with:
        \\
        \\    {s}
        \\
        \\then point `.{s}` in InitOptions at it. See "Model selection" in USAGE.md.
    , .{ what, path, download_cmd, option });
}

fn getModelPath(
    allocator: Allocator,
    io: std.Io,
    exe_relative_path: []const u8,
    bundle_relative_path: []const u8,
) ![:0]const u8 {
    const NSBundle = objc.getClass("NSBundle") orelse return error.ObjCClassNotFound;
    const mainBundle_sel = objc.Sel.registerName("mainBundle");
    const resourcePath_sel = objc.Sel.registerName("resourcePath");
    const utf8_sel = objc.Sel.registerName("UTF8String");

    const bundle = NSBundle.msgSend(Object, mainBundle_sel, .{});
    if (bundle.value != 0) {
        const resource_path_ns = bundle.msgSend(Object, resourcePath_sel, .{});
        if (resource_path_ns.value != 0) {
            const resource_path = resource_path_ns.msgSend([*:0]const u8, utf8_sel, .{});
            const bundle_path = try std.fmt.allocPrintSentinel(
                allocator,
                "{s}/{s}",
                .{ resource_path, bundle_relative_path },
                0,
            );

            if (std.Io.Dir.accessAbsolute(io, bundle_path, .{})) |_| {
                return bundle_path;
            } else |_| {
                return try getExeRelativePath(allocator, io, exe_relative_path);
            }
        }
    }

    return try getExeRelativePath(allocator, io, exe_relative_path);
}

/// Returns the `.mlmodelc` beside `path` when `path` is an `.mlpackage` and one is there,
/// and `path` itself otherwise.
///
/// A package has to be compiled before it can be loaded, into a temporary directory that
/// is different in every process. The compile is quick; what it costs is the load after
/// it, because the OS caches its Neural Engine specialization by model path and a fresh
/// path never hits. Measured: 2.8s to load from the package, 60ms from a compiled model
/// that has been loaded from the same place before.
fn preferPrecompiled(allocator: Allocator, io: std.Io, path: [:0]const u8) ![:0]const u8 {
    const package_ext = ".mlpackage";
    if (!std.mem.endsWith(u8, path, package_ext)) return path;
    const compiled = try std.fmt.allocPrintSentinel(
        allocator,
        "{s}.mlmodelc",
        .{path[0 .. path.len - package_ext.len]},
        0,
    );
    std.Io.Dir.accessAbsolute(io, compiled, .{}) catch return path;
    return compiled;
}

fn getExeRelativePath(allocator: Allocator, io: std.Io, relative_path: []const u8) ![:0]const u8 {
    var exe_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const exe_path_len = std.process.executableDirPath(io, &exe_path_buf) catch |err| {
        std.log.err("Failed to get executable path: {}", .{err});
        return error.ExePathFailed;
    };
    const exe_path = exe_path_buf[0..exe_path_len];

    // Try exe-relative path first
    const exe_relative = try std.fmt.allocPrintSentinel(
        allocator,
        "{s}/../{s}",
        .{ exe_path, relative_path },
        0,
    );
    std.Io.Dir.accessAbsolute(io, exe_relative, .{}) catch {
        // This is the case where we are testing, files are in a different place.
        allocator.free(exe_relative);
        const cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator);
        defer allocator.free(cwd);
        return try std.fmt.allocPrintSentinel(
            allocator,
            "{s}/zig-out/{s}",
            .{ cwd, relative_path },
            0,
        );
    };
    return exe_relative;
}

pub const NLEmbedder = struct {
    embedder_obj: Object,
    mutex: Mutex,
    io: std.Io,

    pub const VEC_SZ = 512;
    pub const VEC_TYPE = f32;
    pub const ID = EmbeddingModel.apple_nlembedding;
    pub const REFERENCE_IMPLEMENTATION_NAME = "apple-nlembedding";
    pub const THRESHOLD = 0.40;
    pub const STRICT_THRESHOLD = THRESHOLD * 2;
    // Suffixed by storage type: a quantized database is not loadable by an unquantized
    // build, so the two live side by side rather than one refusing the other's file.
    pub const PATH = @tagName(ID) ++ db_suffix ++ ".db";

    pub fn init(io: std.Io) !NLEmbedder {
        const init_zone = tracy.beginZone(@src(), .{ .name = "embed.zig:init" });
        defer init_zone.end();
        const pool = objc.AutoreleasePool.init();
        defer pool.deinit();

        var NSString = objc.getClass("NSString").?;
        var NLEmbedding = objc.getClass("NLEmbedding").?;
        const fromUTF8 = objc.Sel.registerName("stringWithUTF8String:");

        const sentenceEmbeddingForLang = objc.Sel.registerName("sentenceEmbeddingForLanguage:");
        const language = "en";
        const ns_lang = NSString.msgSend(Object, fromUTF8, .{language});

        const embedder_obj = NLEmbedding.msgSend(Object, sentenceEmbeddingForLang, .{ns_lang});
        if (embedder_obj.value == 0) {
            std.log.err("NLEmbedding.sentenceEmbeddingForLanguage returned nil - ensure this is called from main thread", .{});
            return error.EmbedderInitFailed;
        }
        assert(embedder_obj.getProperty(c_int, "dimension") == VEC_SZ);

        return .{
            .embedder_obj = embedder_obj.retain(),
            .mutex = .init,
            .io = io,
        };
    }

    pub fn init_self(self: *NLEmbedder, io: std.Io) !void {
        const obj = try NLEmbedder.init(io);
        self.embedder_obj = obj.embedder_obj;
        self.mutex = obj.mutex;
        self.io = obj.io;
    }

    pub fn embedder(self: *NLEmbedder) Embedder {
        return .{
            .ptr = self,
            .splitFn = split,
            .embedFn = embed,
            .embedBatchFn = sequentialEmbedBatch(embed),
            .deinitFn = deinitFn,
            .id = ID,
            .threshold = THRESHOLD,
            .strict_threshold = STRICT_THRESHOLD,
            .path = PATH,
        };
    }

    pub fn deinit(self: *NLEmbedder) void {
        self.embedder_obj.release();
    }

    fn deinitFn(ptr: *anyopaque) void {
        const self: *NLEmbedder = @ptrCast(@alignCast(ptr));
        self.deinit();
    }

    fn split(self: *anyopaque, note: []const u8) SentenceSpliterator {
        _ = self;
        return SentenceSpliterator.init(note);
    }

    fn embed(
        ptr: *anyopaque,
        allocator: Allocator,
        str: []const u8,
    ) !?EmbeddingModelOutput {
        const self: *NLEmbedder = @ptrCast(@alignCast(ptr));
        const zone = tracy.beginZone(@src(), .{ .name = "embed.zig:embed" });
        defer zone.end();
        const pool = objc.AutoreleasePool.init();
        defer pool.deinit();

        var NSString = objc.getClass("NSString").?;
        const fromUTF8 = objc.Sel.registerName("stringWithUTF8String:");
        const getVectorForString = objc.Sel.registerName("getVector:forString:");

        if (str.len == 0 or str[0] == 0) {
            std.log.info("Skipping embed of zero-length string", .{});
            return null;
        }
        if (!isAlphanumeric(str[0]) or !isAlphanumeric(str[str.len - 1])) {
            std.log.warn("Embedding str with punctuation is likely unexpected -> '{s}'", .{str});
        }

        const c_str = try std.fmt.allocPrintSentinel(allocator, "{s}", .{str}, 0);
        defer allocator.free(c_str);
        const objc_str = NSString.msgSend(Object, fromUTF8, .{c_str.ptr});
        // defer objc_str.release();

        const VecType = @Vector(VEC_SZ, VEC_TYPE);
        const vec_buf: [*]align(@alignOf(VecType)) VEC_TYPE = @ptrCast((try allocator.alignedAlloc(
            VEC_TYPE,
            std.mem.Alignment.of(VecType),
            VEC_SZ,
        )).ptr);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.embedder_obj.msgSend(bool, getVectorForString, .{ vec_buf, objc_str })) {
            std.log.warn("Failed to embed '{s}'", .{str[0..@min(str.len, 10)]});
            return null;
        }

        // NLEmbedding does not return unit vectors; normalize to enable dot-product search.
        const vec_ptr: *VecType = @ptrCast(vec_buf);
        const mag: VecType = @splat(@sqrt(@reduce(.Add, vec_ptr.* * vec_ptr.*)));
        vec_ptr.* = vec_ptr.* / mag;

        return EmbeddingModelOutput{
            .apple_nlembedding = @ptrCast(vec_buf),
        };
    }
};

pub const LlamaNomicEmbedTextV15F32 = struct {
    pub const VEC_SZ = 768;
    pub const VEC_TYPE = f32;
    pub const ID = EmbeddingModel.llama_nomic_embed_text_v1_5_f32;
    pub const REFERENCE_IMPLEMENTATION_NAME = "llama-nomic-embed-text-v1-5-f32";
    // Tuned against src/benchmark.zig, which scores both matches and non-matches
    // through this cutoff: 0.55 peaks at 89.6%, against 81.7% at 0.45 and 79.1%
    // at 0.65. nomic-embed's similarities sit higher than mpnet's, so mpnet's
    // 0.40 let far too much through.
    pub const THRESHOLD = 0.55;
    pub const STRICT_THRESHOLD = THRESHOLD + 0.1;
    // Suffixed by storage type: a quantized database is not loadable by an unquantized
    // build, so the two live side by side rather than one refusing the other's file.
    pub const PATH = @tagName(ID) ++ db_suffix ++ ".db";

    pub const MODEL_FILE = "nomic-embed-text-v1.5.f32.gguf";
    pub const MODEL_PATH = "share/" ++ MODEL_FILE;
    pub const BUNDLE_MODEL_PATH = MODEL_FILE;
    /// Fetches the gguf into the current directory. Keep in step with build.zig and USAGE.md.
    pub const DOWNLOAD_CMD = "curl -LO https://huggingface.co/nomic-ai/" ++
        "nomic-embed-text-v1.5-GGUF/resolve/main/" ++ MODEL_FILE;

    pub const MAX_CTX = 8192;

    pub const InitOptions = struct {
        /// Path to the `.gguf`. Null takes the DVE_LLAMA_MODEL environment variable, then
        /// looks for `BUNDLE_MODEL_PATH` in the app bundle's resources, then for
        /// `MODEL_PATH` beside the executable.
        model_path: ?[]const u8 = null,
    };

    pub fn init(io: std.Io, opts: InitOptions) !LlamaNomicEmbedTextV15F32 {
        if (comptime !llama.enabled) return llama.Error.LlamaNotLinked;

        // Nothing here outlives init: the bridge keeps its own copy of the path.
        var path_buf: [4 * std.Io.Dir.max_path_bytes]u8 = undefined;
        var path_alloc = std.heap.FixedBufferAllocator.init(&path_buf);
        const model_path_z: [:0]const u8 =
            if (opts.model_path orelse envVar("DVE_LLAMA_MODEL")) |p|
                path_alloc.allocator().dupeSentinel(u8, p, 0) catch return error.NameTooLong
            else
                getModelPath(path_alloc.allocator(), io, MODEL_PATH, BUNDLE_MODEL_PATH) catch
                    return error.PathAllocFailed;
        std.Io.Dir.cwd().access(io, model_path_z, .{}) catch {
            logModelNotFound("llama model", model_path_z, DOWNLOAD_CMD, "model_path");
            return error.ModelNotFound;
        };

        // The model lives in the bridge, one per process, rather than in this
        // struct. Loading it here is what makes a bad path fail at init
        // instead of on the first embed.
        try llama.load(model_path_z);
        return .{};
    }

    pub fn init_self(self: *LlamaNomicEmbedTextV15F32, io: std.Io, opts: InitOptions) !void {
        self.* = try LlamaNomicEmbedTextV15F32.init(io, opts);
    }

    pub fn deinit(self: *LlamaNomicEmbedTextV15F32) void {
        _ = self;
        return;
    }

    pub fn embedder(self: *LlamaNomicEmbedTextV15F32) Embedder {
        return .{
            .ptr = self,
            .splitFn = split,
            .embedFn = embed,
            .embedBatchFn = embedBatch,
            .deinitFn = deinitFn,
            .id = ID,
            .threshold = THRESHOLD,
            .strict_threshold = STRICT_THRESHOLD,
            .path = PATH,
        };
    }

    fn deinitFn(ptr: *anyopaque) void {
        _ = ptr;
        return;
    }

    fn split(self: *anyopaque, note: []const u8) SentenceSpliterator {
        _ = self;
        return SentenceSpliterator.init(note);
    }

    fn embed(ptr: *anyopaque, allocator: Allocator, str: []const u8) !?EmbeddingModelOutput {
        _ = ptr;
        const zone = tracy.beginZone(@src(), .{
            .name = "embed.zig:LlamaNomicEmbedTextV15F32.embed",
        });
        defer zone.end();

        if (str.len == 0) {
            std.log.info("Skipping embed of zero-length string", .{});
            return null;
        }
        if (!isAlphanumeric(str[0]) or !isAlphanumeric(str[str.len - 1])) {
            std.log.warn("Embedding str with punctuation is likely unexpected -> '{s}'", .{str});
        }

        // The bridge takes a C string; anything past MAX_CTX is truncated there.
        const c_str = try allocator.dupeSentinel(u8, str, 0);
        defer allocator.free(c_str);

        const VecType = @Vector(VEC_SZ, VEC_TYPE);
        const vec_buf = try allocator.alignedAlloc(
            VEC_TYPE,
            std.mem.Alignment.of(VecType),
            VEC_SZ,
        );
        errdefer allocator.free(vec_buf);

        // llama_bridge.c serializes calls and L2-normalizes the result.
        const written = try llama.embed(vec_buf, c_str);
        assert(written == VEC_SZ);

        return EmbeddingModelOutput{
            .llama_nomic_embed_text_v1_5_f32 = @ptrCast(vec_buf.ptr),
        };
    }

    /// Hands every string to the bridge at once, which packs several sequences
    /// into each llama_decode. For the short chunks dve embeds that is worth
    /// several times the one-at-a-time path, whose cost is dominated by per-
    /// decode dispatch overhead rather than by the model.
    ///
    /// The strings `embed` declines are filtered out here rather than sent and
    /// rejected, because the bridge has no way to say "skipped" for one entry
    /// of a batch. Their slots stay null, so the result still lines up with
    /// `strs` and still matches what `embed` returns for each string alone.
    fn embedBatch(
        ptr: *anyopaque,
        allocator: Allocator,
        strs: []const []const u8,
    ) ![]?EmbeddingModelOutput {
        _ = ptr;
        const zone = tracy.beginZone(@src(), .{
            .name = "embed.zig:LlamaNomicEmbedTextV15F32.embedBatch",
        });
        defer zone.end();

        const outs = try allocator.alloc(?EmbeddingModelOutput, strs.len);
        errdefer allocator.free(outs);
        @memset(outs, null);
        if (strs.len == 0) return outs;

        // Three parallel arrays over the strings that survive filtering: the
        // NUL-terminated copy the bridge reads, the buffer it writes, and the
        // slot in `outs` each one came from.
        const c_strs = try allocator.alloc([*:0]const u8, strs.len);
        const bufs = try allocator.alloc([*]f32, strs.len);
        const slots = try allocator.alloc(usize, strs.len);

        const VecType = @Vector(VEC_SZ, VEC_TYPE);
        var n: usize = 0;
        for (strs, 0..) |str, i| {
            if (str.len == 0) {
                std.log.info("Skipping embed of zero-length string", .{});
                continue;
            }
            if (!isAlphanumeric(str[0]) or !isAlphanumeric(str[str.len - 1])) {
                std.log.warn("Embedding str with punctuation is likely unexpected -> '{s}'", .{str});
            }
            // The bridge takes a C string; anything past MAX_CTX is truncated there.
            c_strs[n] = (try allocator.dupeSentinel(u8, str, 0)).ptr;
            // Each vector is allocated on its own so it carries the alignment
            // @Vector(VEC_SZ, VEC_TYPE) needs. One slab with a VEC_SZ stride
            // would only be aligned at its start.
            bufs[n] = (try allocator.alignedAlloc(
                VEC_TYPE,
                std.mem.Alignment.of(VecType),
                VEC_SZ,
            )).ptr;
            slots[n] = i;
            n += 1;
        }
        if (n == 0) return outs;

        const written = try llama.embedBatch(bufs[0..n], VEC_SZ, c_strs[0..n]);
        assert(written == VEC_SZ);

        for (slots[0..n], bufs[0..n]) |slot, buf| {
            // Storing the buffers as [*]f32 to hand them to the bridge drops the
            // alignment from the type; the alignedAlloc above is what makes the
            // @alignCast true.
            outs[slot] = EmbeddingModelOutput{
                .llama_nomic_embed_text_v1_5_f32 = @ptrCast(@alignCast(buf)),
            };
        }
        return outs;
    }
};

pub const Chunk = struct {
    contents: []const u8,
    start_i: u32,
    end_i: u32,
    type: Type,

    pub const Type = enum { string, url };

    pub fn strip(self: *Chunk) void {
        for (self.contents, 0..) |c, i| {
            if (isAlphanumeric(c)) {
                self.contents = self.contents[i..];
                break;
            }
            self.start_i += 1;
        }
        for (1..self.contents.len + 1) |neg_i| {
            const i = self.contents.len - neg_i;
            const c = self.contents[i];
            if (isAlphanumeric(c)) {
                self.contents = self.contents[0 .. i + 1];
                break;
            }
            if (self.end_i <= self.start_i) {
                self.contents = self.contents[0..0];
                return;
            }
            self.end_i -= 1;
        }
        assert(self.end_i >= self.start_i);
        assert(self.end_i - self.start_i == self.contents.len);
    }
};

pub fn Spliterator(comptime delimiters: []const u8) type {
    return struct {
        buffer: []const u8,
        index: usize,

        const Self = @This();
        const url_prefixes = [_][]const u8{ "https://", "http://" };

        fn isDelimiter(c: u8) bool {
            for (delimiters) |d| {
                if (c == d) return true;
            }
            return false;
        }

        fn isUrlEnd(c: u8) bool {
            return c == ' ' or c == '\n' or c == '\t';
        }

        fn findUrlPrefix(buf: []const u8) ?usize {
            for (url_prefixes) |prefix| {
                if (std.mem.startsWith(u8, buf, prefix)) return prefix.len;
            }
            return null;
        }

        pub fn init(buffer: []const u8) Self {
            return .{
                .buffer = buffer,
                .index = 0,
            };
        }

        pub fn next(self: *Self) ?Chunk {
            while (true) {
                // We are done
                if (self.index >= self.buffer.len) return null;

                // Skip past garbage characters
                while (self.index < self.buffer.len and isDelimiter(self.buffer[self.index])) {
                    self.index += 1;
                }

                // We are done
                if (self.index >= self.buffer.len) return null;

                if (findUrlPrefix(self.buffer[self.index..])) |_| {
                    // There is a URL ahead
                    const start = self.index;
                    while (self.index < self.buffer.len and !isUrlEnd(self.buffer[self.index])) {
                        self.index += 1;
                    }
                    const contents = self.buffer[start..self.index];
                    var out = Chunk{
                        .contents = contents,
                        .start_i = @intCast(start),
                        .end_i = @intCast(self.index),
                        .type = .url,
                    };
                    out.strip();
                    if (out.contents.len == 0) continue; // The contents are junk, skip this block
                    return out;
                }

                // We are at the first non-garbage character
                const start = self.index;
                while (self.index < self.buffer.len and !isDelimiter(self.buffer[self.index])) {
                    if (findUrlPrefix(self.buffer[self.index..])) |_| break;
                    self.index += 1;
                }

                // We found a delimiter or the start of a URL
                var end = self.index;
                while (end > start and self.buffer[end - 1] == ' ') {
                    end -= 1;
                }

                if (end == start) return self.next();

                const contents = self.buffer[start..end];
                var out = Chunk{
                    .contents = contents,
                    .start_i = @intCast(start),
                    .end_i = @intCast(end),
                    .type = .string,
                };
                out.strip();
                if (out.contents.len == 0) continue; // The contents are junk, skip this block
                return out;
            }
        }

        pub fn collectAll(self: *Self, allocator: Allocator) ![]Chunk {
            var list: std.ArrayList(Chunk) = .empty;
            errdefer list.deinit(allocator);
            while (self.next()) |chunk| {
                try list.append(allocator, chunk);
            }
            return list.toOwnedSlice(allocator);
        }
    };
}

const WORD_SPLIT_DELIMITERS = ".!?\n, ();\":";
pub const WordSpliterator = Spliterator(WORD_SPLIT_DELIMITERS);
const SENTENCE_SPLIT_DELIMITERS = ".!?\n";
pub const SentenceSpliterator = Spliterator(SENTENCE_SPLIT_DELIMITERS);

test "spliterator - strip" {
    const ResultType = struct { contents: []const u8, type: Chunk.Type };
    const cases = [_]struct {
        input: []const u8,
        expected: []const ResultType,
    }{
        .{
            .input = " foo       bar\t\t. baz .",
            .expected = &[_]ResultType{
                .{ .contents = "foo       bar", .type = .string },
                .{ .contents = "baz", .type = .string },
            },
        },
        .{
            .input = "@ #$%^&*()_+<>:\"{}|,/;'[]\\foo@ #$%^&*()_+<>:\"{}|,/;'[]\\",
            .expected = &[_]ResultType{
                .{ .contents = "foo", .type = .string },
            },
        },
        .{
            .input = "skip.******.garbage.",
            .expected = &[_]ResultType{
                .{ .contents = "skip", .type = .string },
                .{ .contents = "garbage", .type = .string },
            },
        },
    };

    for (cases) |case| {
        var splitter = SentenceSpliterator.init(case.input);
        const chunks = try splitter.collectAll(std.testing.allocator);
        defer std.testing.allocator.free(chunks);

        try expectEqual(case.expected.len, chunks.len);
        for (case.expected, chunks) |expected, chunk| {
            try expectEqualStrings(expected.contents, case.input[chunk.start_i..chunk.end_i]);
            try expectEqualStrings(expected.contents, chunk.contents);
            try expectEqual(expected.type, chunk.type);
        }
    }
}

test "spliterator - does not split on delimiters inside URLs" {
    const ResultType = struct { contents: []const u8, type: Chunk.Type };
    const cases = [_]struct {
        input: []const u8,
        expected: []const ResultType,
    }{
        .{
            .input = "foo https://google.com",
            .expected = &[_]ResultType{
                .{ .contents = "foo", .type = .string },
                .{ .contents = "https://google.com", .type = .url },
            },
        },
        .{
            .input = "foo http://foobar.net",
            .expected = &[_]ResultType{
                .{ .contents = "foo", .type = .string },
                .{ .contents = "http://foobar.net", .type = .url },
            },
        },
        .{
            .input = "foo https://en.wikipedia.org/wiki/Dog",
            .expected = &[_]ResultType{
                .{ .contents = "foo", .type = .string },
                .{ .contents = "https://en.wikipedia.org/wiki/Dog", .type = .url },
            },
        },
        .{
            .input = "one https://en.wikipedia.org/wiki/Dog\n2 https://en.wikipedia.org/wiki/Cat",
            .expected = &[_]ResultType{
                .{ .contents = "one", .type = .string },
                .{ .contents = "https://en.wikipedia.org/wiki/Dog", .type = .url },
                .{ .contents = "2", .type = .string },
                .{ .contents = "https://en.wikipedia.org/wiki/Cat", .type = .url },
            },
        },
    };

    for (cases) |case| {
        var splitter = SentenceSpliterator.init(case.input);
        const chunks = try splitter.collectAll(std.testing.allocator);
        defer std.testing.allocator.free(chunks);

        for (case.expected, chunks) |expected, chunk| {
            try expectEqualStrings(expected.contents, chunk.contents);
            try expectEqual(expected.type, chunk.type);
        }
    }
}

test "embed - nlembed solo" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var nl = try NLEmbedder.init(std.testing.io);
    defer nl.deinit();

    var e = nl.embedder();

    var output = try e.embed(allocator, "Hello world");
    // We don't check this too hard because the work to save vectors is not worth the reward.
    // Better to check at the interface level. i.e. we don't care what the specific embedding is
    // as long as the output is what we desire. This test is just to verify that the embedder
    // doesn't do anything FUBAR.
    var vec = output.?.apple_nlembedding.*;
    var sum = @reduce(.Add, vec);
    try std.testing.expectApproxEqAbs(0.08761532, sum, 1e-4);

    output = try e.embed(allocator, "Hello again world");
    vec = output.?.apple_nlembedding.*;
    sum = @reduce(.Add, vec);
    try std.testing.expectApproxEqAbs(0.83664304, sum, 1e-4);
}

test "embed - mpnetembed init with autorelease pool (simulates Swift caller)" {
    // Swift has an autorelease pool on the calling thread. If init() over-releases
    // autoreleased objects, the pool drain will crash with EXC_BAD_ACCESS.
    const pool = objc.AutoreleasePool.init();
    var mpnet = try MpnetEmbedder.init(std.testing.io, .{});
    pool.deinit(); // drains the pool — crashes here if double-release
    defer mpnet.deinit();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var e = mpnet.embedder();
    const output = try e.embed(arena.allocator(), "Hello world");
    try std.testing.expect(output != null);
}

test "embed - nlembedder init with autorelease pool (simulates Swift caller)" {
    // Swift has an autorelease pool on the calling thread. If init() over-releases
    // autoreleased objects, the pool drain will crash with EXC_BAD_ACCESS.
    const pool = objc.AutoreleasePool.init();
    var nlembed = try NLEmbedder.init(std.testing.io);
    pool.deinit(); // drains the pool — crashes here if double-release
    defer nlembed.deinit();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var e = nlembed.embedder();
    const output = try e.embed(arena.allocator(), "Hello world");
    try std.testing.expect(output != null);
}

test "embed - mpnetembed solo" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var mpnet = try MpnetEmbedder.init(std.testing.io, .{});
    defer mpnet.deinit();

    var e = mpnet.embedder();

    const output = try e.embed(allocator, "Hello world");
    try std.testing.expect(output != null);

    const vec = output.?.mpnet_embedding.*;
    const vec_array: [768]f32 = vec;

    // Reference values come from the Python implementation, which runs in fp32. The shipped
    // CoreML model is fp16, because that is the only precision the Neural Engine accepts, and
    // fp16 weights cost about 1e-3 per component here -- measured 7.7e-4, 1.4e-4 and 3.6e-4
    // on these three, and 2.0e-3 on the sum of all 768. Direction survives: cosine against
    // the fp32 conversion is 0.999982, and every group in src/benchmark.zig scores identically.
    //
    // So these bounds are sized to catch a model that is *wrong*, not one that is imprecise.
    // They would not have let the first fp16 attempt through: it overflowed the attention mask
    // and returned vectors unrelated to these.
    for (&[_]f32{ 2.6249737e-2, 1.3395556e-2, -4.533195e-3 }, vec_array[0..3]) |exp, got| {
        try std.testing.expectApproxEqAbs(exp, got, 2e-3);
    }
    const sum = @reduce(.Add, vec);
    try std.testing.expectApproxEqAbs(-2.155769e-1, sum, 5e-3);

    // The guard that does not care about precision at all, and the one that actually caught
    // the broken conversion: related text has to score above unrelated text. The first fp16
    // attempt scored dog~airplane at 1.000.
    const dog = (try e.embed(allocator, "dog")).?.mpnet_embedding.*;
    const puppy = (try e.embed(allocator, "puppy")).?.mpnet_embedding.*;
    const airplane = (try e.embed(allocator, "airplane")).?.mpnet_embedding.*;
    const near = @reduce(.Add, dog * puppy);
    const far = @reduce(.Add, dog * airplane);
    try std.testing.expect(near > far + 0.2);
}

test "embed skip empty" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var nl = try NLEmbedder.init(std.testing.io);
    defer nl.deinit();

    var e = nl.embedder();

    try expectEqual(null, try e.embed(allocator, ""));
}

test "embed skip failures" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var nl = try NLEmbedder.init(std.testing.io);
    defer nl.deinit();

    var e = nl.embedder();

    _ = (try e.embed(allocator, "4327897493287498(*^(*&(# FKJDHDHLKDJHL")).?;
}

test "embed - nlembed thread safety" {
    var nl = try NLEmbedder.init(std.testing.io);
    defer nl.deinit();
    var e = nl.embedder();
    try threadSafetyTest(&e);
}

test "embed - mpnetembed thread safety" {
    var mpnet = try MpnetEmbedder.init(std.testing.io, .{});
    defer mpnet.deinit();
    var e = mpnet.embedder();
    try threadSafetyTest(&e);
}

test "embedding reference implementation" {
    if (true) return error.SkipZigTest;
    const phrases = [_][]const u8{
        "dog",
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models convert text into vector representations",
        "Zig is a systems programming language designed for correctness",
    };

    const cases = [_]EmbeddingModel{
        .mpnet_embedding,
    };

    const script_path = "models/embed_phrase.py";
    const python_path = "models/venv/bin/python";

    for (cases) |model| {
        var mpnet: MpnetEmbedder = undefined;
        var nlembed: NLEmbedder = undefined;
        var e: Embedder = switch (model) {
            .mpnet_embedding => blk: {
                mpnet = try MpnetEmbedder.init(std.testing.io, .{});
                break :blk mpnet.embedder();
            },
            .apple_nlembedding => blk: {
                nlembed = try NLEmbedder.init(std.testing.io);
                break :blk nlembed.embedder();
            },
        };
        defer e.deinit();

        for (phrases) |phrase| {
            // Get reference embedding from Python
            const exec = try std.process.run(std.testing.allocator, std.testing.io, .{
                .argv = &.{ python_path, script_path, model.referenceImplementationName(), phrase },
            });
            defer std.testing.allocator.free(exec.stdout);
            defer std.testing.allocator.free(exec.stderr);

            if (!exec.term.success()) {
                std.debug.print("Python script failed ({any}):\n{s}\n", .{ exec.term, exec.stderr });
                return error.PythonScriptFailed;
            }

            // Get Zig embedding
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const output = (try e.embed(arena.allocator(), phrase)) orelse return error.EmbedReturnedNull;
            const zig_slice = output.slice();
            const vec_sz = output.vecSize();

            // Parse the reference embedding (one float per line)
            var ref_vec: [MpnetEmbedder.VEC_SZ]f32 = undefined;
            var line_iter = std.mem.splitScalar(u8, exec.stdout, '\n');
            var idx: usize = 0;
            while (line_iter.next()) |line| {
                if (line.len == 0) continue;
                ref_vec[idx] = std.fmt.parseFloat(f32, line) catch |err| {
                    std.debug.print("Failed to parse float '{s}': {}\n", .{ line, err });
                    return err;
                };
                idx += 1;
                if (idx >= vec_sz) break;
            }
            try expectEqual(vec_sz, idx);

            // Compare element-by-element with tolerance
            const tolerance: f32 = 1e-4;
            for (0..vec_sz) |i| {
                const diff = @abs(zig_slice[i] - ref_vec[i]);
                if (diff > tolerance) {
                    std.debug.print(
                        "Mismatch for phrase \"{s}\" at index {d}: zig={e:.8} ref={e:.8} diff={e:.8}\n",
                        .{ phrase, i, zig_slice[i], ref_vec[i], diff },
                    );
                    return error.TestExpectedEqual;
                }
            }
        }
    }
}

fn threadSafetyTest(e: *Embedder) !void {
    const n_threads = 4;
    const n_iters = 50;
    const inputs = [_][]const u8{
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Zig is a systems programming language",
    };

    var barrier: std.Io.Event = .unset;
    var threads: [n_threads]std.Thread = undefined;
    for (&threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn run(embedder: *Embedder, input: []const u8, b: *std.Io.Event) void {
                b.waitUncancelable(std.testing.io);
                for (0..n_iters) |_| {
                    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                    defer arena.deinit();
                    const result = embedder.embed(arena.allocator(), input) catch |err| {
                        std.debug.panic("embed failed: {}", .{err});
                    };
                    std.debug.assert(result != null);
                }
            }
        }.run, .{ e, inputs[i], &barrier });
    }
    barrier.set(std.testing.io);
    for (&threads) |*t| t.join();
}

// Reloading the model used to abort inside MetalPerformanceShadersGraph about one run in
// eight, somewhere in whichever mpnet test happened to be running:
//
//     MPSGraphTensorData.mm:223: failed assertion `shape.count = 0 != strides.count = 3'
//
// `MLModel.compileModelAtURL` writes its output to a temporary directory the system may
// reclaim, and `init` compiled afresh every time and never moved it. Loading a precompiled
// `.mlmodelc` instead never failed in 28 runs, against 6 failures in 40 runs that compiled,
// which is what pinned it to the compile step; concurrency turned out to be irrelevant.
//
// `ModelCache` compiles once per process, so this is now the cheap test it looks like. If
// per-init compilation ever comes back, this is where it will show up -- and it will show up
// as an abort rather than a failure, so a flake here means that, not a bad assertion.
test "embed - mpnetembed survives repeated reload" {
    for (0..40) |_| {
        var mpnet = try MpnetEmbedder.init(std.testing.io, .{});
        defer mpnet.deinit();
        var e = mpnet.embedder();

        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const out = try e.embed(arena.allocator(), "Hello world");
        try std.testing.expect(out != null);
    }
}

test "embed - output is L2-normalized (mpnet)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var mpnet = try MpnetEmbedder.init(std.testing.io, .{});
    defer mpnet.deinit();
    var e = mpnet.embedder();

    const phrases = [_][]const u8{
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models convert text into vector representations",
        "Zig is a systems programming language designed for correctness",
        "a",
    };
    for (phrases) |phrase| {
        const output = (try e.embed(arena.allocator(), phrase)) orelse return error.EmbedReturnedNull;
        try validateL2(MpnetEmbedder.VEC_SZ, MpnetEmbedder.VEC_TYPE, output.mpnet_embedding.*);
    }
}

test "embed - output is L2-normalized (nlembed)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var nl = try NLEmbedder.init(std.testing.io);
    defer nl.deinit();
    var e = nl.embedder();

    const phrases = [_][]const u8{
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models convert text into vector representations",
        "Zig is a systems programming language designed for correctness",
        "a",
    };
    for (phrases) |phrase| {
        const output = (try e.embed(arena.allocator(), phrase)) orelse return error.EmbedReturnedNull;
        try validateL2(NLEmbedder.VEC_SZ, NLEmbedder.VEC_TYPE, output.apple_nlembedding.*);
    }
}

// This is technically valid but likely unexpected. Throw a warning instead of failing.
test "embed with punctuation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var nl = try NLEmbedder.init(std.testing.io);
    defer nl.deinit();
    var mpnet = try MpnetEmbedder.init(std.testing.io, .{});
    defer mpnet.deinit();
    var e1 = nl.embedder();
    var e2 = nl.embedder();

    _ = try e1.embed(allocator, "*foo bar*");
    _ = try e2.embed(allocator, "*foo bar*");
}

test "embed - LlamaNomicEmbedTextV15F32 solo" {
    // The bridge is only compiled and linked for -Dembedding-model=llama_...
    if (!llama.enabled) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var nl = try LlamaNomicEmbedTextV15F32.init(std.testing.io, .{});
    defer nl.deinit();

    var e = nl.embedder();

    var output = try e.embed(allocator, "Hello world");
    var vec = output.?.llama_nomic_embed_text_v1_5_f32.*;
    var sum = @reduce(.Add, vec);
    try std.testing.expectApproxEqAbs(0.23469175, sum, 1e-4);

    output = try e.embed(allocator, "Hello again world");
    vec = output.?.llama_nomic_embed_text_v1_5_f32.*;
    sum = @reduce(.Add, vec);
    try std.testing.expectApproxEqAbs(0.17591982, sum, 1e-4);
}

/// Checks `e.embedBatch(strs)` against `e.embed` run on each string alone.
///
/// `tolerance` is the largest per-component difference allowed. Backends whose
/// batch is `sequentialEmbedBatch` make the identical calls either way and must
/// match exactly, so they pass 0. A backend with a real batch does not: packing
/// several sequences into one decode changes the shape of the matmuls, and the
/// GPU's reductions come out in a different order. What must survive is the
/// vector's direction, so the cosine is checked too -- sequences bleeding into
/// each other through a shared decode would still be near-unit-length and would
/// still pass a loose per-component bound, but would not stay parallel.
fn expectBatchMatchesSingles(e: *Embedder, strs: []const []const u8, tolerance: f32) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const batch = try e.embedBatch(allocator, strs);
    try expectEqual(strs.len, batch.len);
    for (strs, batch) |str, batch_out| {
        const single_out = try e.embed(allocator, str);
        if (single_out == null) {
            try expectEqual(null, batch_out);
            continue;
        }
        try expectEqual(std.meta.activeTag(single_out.?), std.meta.activeTag(batch_out.?));

        const single = single_out.?.slice();
        const batched = batch_out.?.slice();
        if (tolerance == 0) {
            try expectEqualSlices(f32, single, batched);
            continue;
        }
        try expectEqual(single.len, batched.len);
        var dot: f64 = 0;
        for (single, batched) |a, b| {
            try std.testing.expectApproxEqAbs(a, b, tolerance);
            dot += @as(f64, a) * @as(f64, b);
        }
        if (dot < 0.9999) {
            std.debug.print("batch and single diverged for '{s}': cosine {d}\n", .{ str, dot });
            return error.TestExpectedApproxEqAbs;
        }
    }
}

const batch_phrases = [_][]const u8{
    "Hello world",
    "",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models convert text into vector representations",
    "",
    "Zig is a systems programming language designed for correctness",
    "a",
};

test "embedBatch - nlembed matches embed" {
    var nl = try NLEmbedder.init(std.testing.io);
    defer nl.deinit();
    var e = nl.embedder();
    try expectBatchMatchesSingles(&e, &batch_phrases, 0);
}

// Measured over 9,480 Wikipedia sentences: max |delta| 1.0e-3, min cosine 0.99999. A model
// with no batch functions embeds the batch one string at a time and matches exactly.
const mpnet_batch_tolerance = 2e-3;

test "embedBatch - mpnetembed matches embed" {
    var mpnet = try MpnetEmbedder.init(std.testing.io, .{});
    defer mpnet.deinit();
    var e = mpnet.embedder();
    try expectBatchMatchesSingles(&e, &batch_phrases, mpnet_batch_tolerance);
}

// Enough strings of each length to reach every path through the batch: full and partial
// predictions of both batch shapes, a tail too short to be worth one, and a string too long
// for either. A vector filed under the wrong string shows up as a cosine far below 1.
test "embedBatch - mpnetembed spans every shape" {
    var mpnet = try MpnetEmbedder.init(std.testing.io, .{});
    defer mpnet.deinit();
    var e = mpnet.embedder();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const subjects = [_][]const u8{ "river", "engine", "violin", "glacier", "market", "falcon", "harbor" };
    var strs: std.ArrayList([]const u8) = .empty;
    // 70 short: two full predictions of the 16-token shape and a partial one.
    for (0..70) |i| {
        try strs.append(allocator, try std.fmt.allocPrint(
            allocator,
            "The {s} number {d} is here",
            .{ subjects[i % subjects.len], i },
        ));
    }
    // 35 medium: one full prediction of the 32-token shape and a tail of three.
    for (0..35) |i| {
        try strs.append(allocator, try std.fmt.allocPrint(
            allocator,
            "Every {s} that was counted on day {d} of the long survey turned out to be " ++
                "older than the people counting it had first believed",
            .{ subjects[i % subjects.len], i },
        ));
    }
    try strs.append(allocator, "");
    try strs.append(allocator, "The archive held letters from the harbor master, ledgers of every " ++
        "ship that had docked in forty years, and a long report on the falcon that nested " ++
        "in the lighthouse, which nobody had asked for and nobody had read until now");
    try expectBatchMatchesSingles(&e, strs.items, mpnet_batch_tolerance);
}

test "embedBatch - empty batch" {
    var nl = try NLEmbedder.init(std.testing.io);
    defer nl.deinit();
    var e = nl.embedder();
    const out = try e.embedBatch(std.testing.allocator, &.{});
    defer std.testing.allocator.free(out);
    try expectEqual(0, out.len);

    if (!llama.enabled) return;
    var ll = try LlamaNomicEmbedTextV15F32.init(std.testing.io, .{});
    defer ll.deinit();
    var le = ll.embedder();
    const llama_out = try le.embedBatch(std.testing.allocator, &.{});
    defer std.testing.allocator.free(llama_out);
    try expectEqual(0, llama_out.len);
}

/// Measured worst case over 3,037 real corpus sentences is 5.4e-4 per component
/// at a cosine of 0.99999 (experiments/embedbench --verify). This bound is loose
/// enough for that and far too tight for a batch that mixed sequences up.
const LLAMA_BATCH_TOLERANCE: f32 = 2e-3;

test "embedBatch - LlamaNomicEmbedTextV15F32 matches embed" {
    if (!llama.enabled) return error.SkipZigTest;

    var ll = try LlamaNomicEmbedTextV15F32.init(std.testing.io, .{});
    defer ll.deinit();
    var e = ll.embedder();
    try expectBatchMatchesSingles(&e, &batch_phrases, LLAMA_BATCH_TOLERANCE);
}

test "embedBatch - LlamaNomicEmbedTextV15F32 all empty" {
    if (!llama.enabled) return error.SkipZigTest;

    var ll = try LlamaNomicEmbedTextV15F32.init(std.testing.io, .{});
    defer ll.deinit();
    var e = ll.embedder();
    try expectBatchMatchesSingles(&e, &.{ "", "" }, LLAMA_BATCH_TOLERANCE);
}

// The bridge packs at most DVE_MAX_SEQ (128) sequences into one llama_decode and
// then starts another, so a batch larger than that exercises the loop that cuts
// one decode from the next. Every string here is distinct, which makes this an
// ordering check as much as a value check: a packing bug that misfiled a vector
// by one shows up as a mismatch rather than as a plausible-looking wrong answer.
test "embedBatch - LlamaNomicEmbedTextV15F32 spans several decodes" {
    if (!llama.enabled) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const n = 300;
    const strs = try arena.allocator().alloc([]const u8, n);
    for (strs, 0..) |*str, i| {
        str.* = try std.fmt.allocPrint(arena.allocator(), "sentence number {d} about badgers", .{i});
    }

    var ll = try LlamaNomicEmbedTextV15F32.init(std.testing.io, .{});
    defer ll.deinit();
    var e = ll.embedder();
    try expectBatchMatchesSingles(&e, strs, LLAMA_BATCH_TOLERANCE);
}

// A batch is cut when either limit is hit, and the token budget is the one a
// document of long paragraphs reaches first. Each of these tokenizes to well
// over a hundred tokens, so the batch is split by tokens rather than by count.
test "embedBatch - LlamaNomicEmbedTextV15F32 splits on the token budget" {
    if (!llama.enabled) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const phrase = "the quick brown fox jumps over the lazy dog ";
    const repeated: [40][phrase.len]u8 = @splat(phrase.*);
    const long: []const u8 = @ptrCast(&repeated);
    const strs = try allocator.alloc([]const u8, 40);
    for (strs, 0..) |*str, i| {
        // Trailing space trimmed: embed warns on non-alphanumeric ends, and 40
        // copies of that warning bury the rest of the test output.
        str.* = try std.fmt.allocPrint(allocator, "{d} {s}", .{ i, long[0 .. long.len - 1] });
    }

    var ll = try LlamaNomicEmbedTextV15F32.init(std.testing.io, .{});
    defer ll.deinit();
    var e = ll.embedder();
    try expectBatchMatchesSingles(&e, strs, LLAMA_BATCH_TOLERANCE);
}

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;
const expectEqualStrings = std.testing.expectEqualStrings;
const expectEqualSlices = std.testing.expectEqualSlices;
const isAlphanumeric = std.ascii.isAlphanumeric;
const parseFromSliceLeaky = std.json.parseFromSliceLeaky;
const process = std.process;
const tokenizer_mod = @import("tokenizer.zig");
const db_suffix = @import("vec_util.zig").db_suffix;
const validateL2 = @import("vec_util.zig").validateL2;
const WordPieceTokenizer = tokenizer_mod.WordPieceTokenizer;
const Mutex = std.Io.Mutex;
const zon = @import("zon");

/// An environment variable of this process, or null. Read through libc, which is always
/// linked here: std hands the environment to `main` now rather than keeping it global, and a
/// library has no `main` to receive it.
fn envVar(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

const llama = @import("llama.zig");

const objc = @import("objc");
const Object = objc.Object;
const tracy = @import("tracy");
