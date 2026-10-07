const std = @import("std");
const dve = @import("dve");
const embed = dve.embed;

const LOG_PATH = "/tmp/dve.log";
var log_fd: ?std.Io.File = null;

/// The `Io` for what this shim does itself: its lock and its log file. Blocking and without a
/// thread pool, which is all either needs, and usable before `dve_init` has run.
const shim_io: std.Io = std.Io.Threaded.global_single_threaded.io();

/// The `Io` the engine runs on, created by `dve_init` and torn down by `dve_deinit`. A C
/// caller has no `Io` to hand over, so this is where one is chosen. It is a real thread pool
/// rather than `shim_io` because the engine splits its code scan across threads.
///
/// For as long as it is live, std keeps no-op handlers installed for SIGIO and SIGPIPE, and
/// puts back whatever was there before on `deinit`.
var engine_threaded: std.Io.Threaded = undefined;

fn logFn(
    comptime message_level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime format: []const u8,
    args: anytype,
) void {
    if (log_fd == null) {
        log_fd = std.Io.Dir.createFileAbsolute(shim_io, LOG_PATH, .{}) catch return;
    }
    var buf: [1024]u8 = undefined;
    const prefix = "[" ++ @tagName(message_level) ++ "] (" ++ @tagName(scope) ++ ") ";
    const msg = std.fmt.bufPrint(&buf, prefix ++ format ++ "\n", args) catch return;
    log_fd.?.writeStreamingAll(shim_io, msg) catch return;
}

pub const std_options: std.Options = .{
    .logFn = logFn,
    .log_level = .debug,
};

const MpnetEmbedder = embed.MpnetEmbedder;
const NLEmbedder = embed.NLEmbedder;

// Both VectorEngine specializations compiled in so the xcframework supports
// runtime model selection. Each has its own on-disk format (768-dim vs 512-dim),
// so a given database directory only works with one model at a time.
const AppleVDB = dve.VectorEngine(.apple_nlembedding);
const MpnetVDB = dve.VectorEngine(.mpnet_embedding);

const ActiveModel = enum { apple_nl, mpnet };

// Global singleton state
var gpa: std.heap.DebugAllocator(.{}) = .init;
var mutex: std.Io.Mutex = .init;
var active_model: ActiveModel = undefined;
var apple_db: ?*AppleVDB = null;
var mpnet_db: ?*MpnetVDB = null;
var initialized = false;

const CError = enum(c_int) {
    Success = 0,
    GenericFail = -1,
    DoubleInit = -2,
    NotInit = -3,
};

export fn dve_init(
    basedir: [*:0]const u8,
    model_path: [*:0]const u8,
    tokenizer_path: [*:0]const u8,
) c_int {
    mutex.lockUncancelable(shim_io);
    defer mutex.unlock(shim_io);

    if (initialized) return @backingInt(CError.DoubleInit);

    const allocator = gpa.allocator();
    const basedir_slice = std.mem.sliceTo(basedir, 0);

    engine_threaded = .init(allocator, .{});
    const io = engine_threaded.io();
    // Undone on every failure below, so a failed init leaves nothing behind.
    var ok = false;
    defer if (!ok) engine_threaded.deinit();

    const dir = std.Io.Dir.openDirAbsolute(io, basedir_slice, .{ .iterate = true }) catch |err| {
        std.log.err("dve_init: failed to open basedir '{s}': {}", .{ basedir_slice, err });
        return @backingInt(CError.GenericFail);
    };

    // Non-empty model_path → mpnet; empty → Apple NL (no model files required).
    const model_slice = std.mem.sliceTo(model_path, 0);
    if (model_slice.len > 0) {
        const tokenizer_slice = std.mem.sliceTo(tokenizer_path, 0);
        mpnet_db = MpnetVDB.init(allocator, io, dir, .{
            .model_path = model_slice,
            .tokenizer_path = if (tokenizer_slice.len > 0) tokenizer_slice else null,
        }) catch |err| {
            std.log.err("dve_init: failed to init VectorEngine: {}", .{err});
            return @backingInt(CError.GenericFail);
        };
        active_model = .mpnet;
    } else {
        apple_db = AppleVDB.init(allocator, io, dir, .{}) catch |err| {
            std.log.err("dve_init: failed to init VectorEngine: {}", .{err});
            return @backingInt(CError.GenericFail);
        };
        active_model = .apple_nl;
    }

    ok = true;
    initialized = true;
    return @backingInt(CError.Success);
}

export fn dve_deinit() c_int {
    mutex.lockUncancelable(shim_io);
    defer mutex.unlock(shim_io);

    if (!initialized) return @backingInt(CError.NotInit);
    switch (active_model) {
        .apple_nl => {
            apple_db.?.deinit();
            apple_db = null;
        },
        .mpnet => {
            mpnet_db.?.deinit();
            mpnet_db = null;
        },
    }
    engine_threaded.deinit();
    initialized = false;
    return @backingInt(CError.Success);
}

export fn dve_embed(key: [*:0]const u8, content: [*:0]const u8) c_int {
    mutex.lockUncancelable(shim_io);
    defer mutex.unlock(shim_io);

    if (!initialized) return @backingInt(CError.NotInit);
    const key_s = std.mem.sliceTo(key, 0);
    const content_s = std.mem.sliceTo(content, 0);
    switch (active_model) {
        .apple_nl => apple_db.?.embedText(key_s, content_s) catch |err| {
            std.log.err("dve_embed: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.embedText(key_s, content_s) catch |err| {
            std.log.err("dve_embed: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
    }
    return @backingInt(CError.Success);
}

export fn dve_embed_async(key: [*:0]const u8, content: [*:0]const u8) c_int {
    // No mutex: embedTextAsync enqueues work and returns immediately.
    // The work queue is thread-safe internally.
    // active_model is set before initialized=true, so reading it here is safe.
    if (!initialized) return @backingInt(CError.NotInit);
    const key_s = std.mem.sliceTo(key, 0);
    const content_s = std.mem.sliceTo(content, 0);
    switch (active_model) {
        .apple_nl => apple_db.?.embedTextAsync(key_s, content_s) catch |err| {
            std.log.err("dve_embed_async: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.embedTextAsync(key_s, content_s) catch |err| {
            std.log.err("dve_embed_async: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
    }
    return @backingInt(CError.Success);
}

export fn dve_search(
    query: [*:0]const u8,
    outbuf: [*c]CDVESearchResult,
    n: u32,
) c_int {
    mutex.lockUncancelable(shim_io);
    defer mutex.unlock(shim_io);

    if (!initialized) return @backingInt(CError.NotInit);

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const query_s = std.mem.sliceTo(query, 0);
    const tmp = arena.allocator().alloc(dve.SearchResult, n) catch {
        return @backingInt(CError.GenericFail);
    };

    const written: usize = switch (active_model) {
        .apple_nl => apple_db.?.search(query_s, tmp) catch |err| {
            std.log.err("dve_search: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.search(query_s, tmp) catch |err| {
            std.log.err("dve_search: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
    };

    for (tmp[0..written], 0..) |sr, i| {
        outbuf[i] = toC(sr);
    }
    return @intCast(written);
}

export fn dve_remove(key: [*:0]const u8) c_int {
    mutex.lockUncancelable(shim_io);
    defer mutex.unlock(shim_io);

    if (!initialized) return @backingInt(CError.NotInit);
    const key_s = std.mem.sliceTo(key, 0);
    switch (active_model) {
        .apple_nl => apple_db.?.removePath(key_s) catch |err| {
            std.log.err("dve_remove: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.removePath(key_s) catch |err| {
            std.log.err("dve_remove: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
    }
    return @backingInt(CError.Success);
}

export fn dve_rename(old_key: [*:0]const u8, new_key: [*:0]const u8) c_int {
    mutex.lockUncancelable(shim_io);
    defer mutex.unlock(shim_io);

    if (!initialized) return @backingInt(CError.NotInit);
    const old_s = std.mem.sliceTo(old_key, 0);
    const new_s = std.mem.sliceTo(new_key, 0);
    switch (active_model) {
        .apple_nl => apple_db.?.renamePath(old_s, new_s) catch |err| {
            std.log.err("dve_rename: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.renamePath(old_s, new_s) catch |err| {
            std.log.err("dve_rename: {}", .{err});
            return @backingInt(CError.GenericFail);
        },
    }
    return @backingInt(CError.Success);
}

// Internal C-compatible result type
const DVE_PATH_MAX = 1024;

const CDVESearchResult = extern struct {
    key: [DVE_PATH_MAX]u8,
    start_i: u32,
    end_i: u32,
    similarity: f32,
};

fn toC(sr: dve.SearchResult) CDVESearchResult {
    var r = CDVESearchResult{
        .key = std.mem.zeroes([DVE_PATH_MAX]u8),
        .start_i = @intCast(sr.start_i),
        .end_i = @intCast(sr.end_i),
        .similarity = sr.similarity,
    };
    const len = @min(sr.path.len, DVE_PATH_MAX - 1);
    @memcpy(r.key[0..len], sr.path[0..len]);
    return r;
}
