const std = @import("std");
const dve = @import("dve");
const embed = dve.embed;

const LOG_PATH = "/tmp/dve.log";
var log_fd: ?std.fs.File = null;

fn logFn(
    comptime message_level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime format: []const u8,
    args: anytype,
) void {
    if (log_fd == null) {
        log_fd = std.fs.createFileAbsolute(LOG_PATH, .{}) catch return;
    }
    var buf: [1024]u8 = undefined;
    const prefix = "[" ++ @tagName(message_level) ++ "] (" ++ @tagName(scope) ++ ") ";
    const msg = std.fmt.bufPrint(&buf, prefix ++ format ++ "\n", args) catch return;
    _ = log_fd.?.write(msg) catch return;
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
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var mutex = std.Thread.Mutex{};
var active_model: ActiveModel = undefined;
var apple_db: ?*AppleVDB = null;
var mpnet_db: ?*MpnetVDB = null;
var apple_embedder: NLEmbedder = undefined;
var mpnet_embedder: MpnetEmbedder = undefined;
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
    mutex.lock();
    defer mutex.unlock();

    if (initialized) return @intFromEnum(CError.DoubleInit);

    const allocator = gpa.allocator();
    const basedir_slice = std.mem.sliceTo(basedir, 0);

    const dir = std.fs.openDirAbsolute(basedir_slice, .{ .iterate = true }) catch |err| {
        std.log.err("dve_init: failed to open basedir '{s}': {}\n", .{ basedir_slice, err });
        return @intFromEnum(CError.GenericFail);
    };

    // Non-empty model_path → mpnet; empty → Apple NL (no model files required).
    const model_slice = std.mem.sliceTo(model_path, 0);
    if (model_slice.len > 0) {
        const tokenizer_slice = std.mem.sliceTo(tokenizer_path, 0);
        mpnet_embedder = MpnetEmbedder.init(.{
            .absolute_model_path = model_slice,
            .absolute_tokenizer_path = if (tokenizer_slice.len > 0) tokenizer_slice else null,
        }) catch |err| {
            std.log.err("dve_init: failed to init MpnetEmbedder: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        };
        mpnet_db = MpnetVDB.init(allocator, dir, mpnet_embedder.embedder()) catch |err| {
            std.log.err("dve_init: failed to init VectorEngine: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        };
        active_model = .mpnet;
    } else {
        apple_embedder = NLEmbedder.init() catch |err| {
            std.log.err("dve_init: failed to init NLEmbedder: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        };
        apple_db = AppleVDB.init(allocator, dir, apple_embedder.embedder()) catch |err| {
            std.log.err("dve_init: failed to init VectorEngine: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        };
        active_model = .apple_nl;
    }

    initialized = true;
    return @intFromEnum(CError.Success);
}

export fn dve_deinit() c_int {
    mutex.lock();
    defer mutex.unlock();

    if (!initialized) return @intFromEnum(CError.NotInit);
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
    initialized = false;
    return @intFromEnum(CError.Success);
}

export fn dve_embed(key: [*:0]const u8, content: [*:0]const u8) c_int {
    mutex.lock();
    defer mutex.unlock();

    if (!initialized) return @intFromEnum(CError.NotInit);
    const key_s = std.mem.sliceTo(key, 0);
    const content_s = std.mem.sliceTo(content, 0);
    switch (active_model) {
        .apple_nl => apple_db.?.embedText(key_s, content_s) catch |err| {
            std.log.err("dve_embed: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.embedText(key_s, content_s) catch |err| {
            std.log.err("dve_embed: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
    }
    return @intFromEnum(CError.Success);
}

export fn dve_embed_async(key: [*:0]const u8, content: [*:0]const u8) c_int {
    // No mutex: embedTextAsync enqueues work and returns immediately.
    // The work queue is thread-safe internally.
    // active_model is set before initialized=true, so reading it here is safe.
    if (!initialized) return @intFromEnum(CError.NotInit);
    const key_s = std.mem.sliceTo(key, 0);
    const content_s = std.mem.sliceTo(content, 0);
    switch (active_model) {
        .apple_nl => apple_db.?.embedTextAsync(key_s, content_s) catch |err| {
            std.log.err("dve_embed_async: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.embedTextAsync(key_s, content_s) catch |err| {
            std.log.err("dve_embed_async: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
    }
    return @intFromEnum(CError.Success);
}

export fn dve_search(
    query: [*:0]const u8,
    outbuf: [*c]CDVESearchResult,
    n: u32,
) c_int {
    mutex.lock();
    defer mutex.unlock();

    if (!initialized) return @intFromEnum(CError.NotInit);

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const query_s = std.mem.sliceTo(query, 0);
    const tmp = arena.allocator().alloc(dve.SearchResult, n) catch {
        return @intFromEnum(CError.GenericFail);
    };

    const written: usize = switch (active_model) {
        .apple_nl => apple_db.?.search(query_s, tmp) catch |err| {
            std.log.err("dve_search: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.search(query_s, tmp) catch |err| {
            std.log.err("dve_search: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
    };

    for (tmp[0..written], 0..) |sr, i| {
        outbuf[i] = toC(sr);
    }
    return @intCast(written);
}

export fn dve_remove(key: [*:0]const u8) c_int {
    mutex.lock();
    defer mutex.unlock();

    if (!initialized) return @intFromEnum(CError.NotInit);
    const key_s = std.mem.sliceTo(key, 0);
    switch (active_model) {
        .apple_nl => apple_db.?.removePath(key_s) catch |err| {
            std.log.err("dve_remove: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.removePath(key_s) catch |err| {
            std.log.err("dve_remove: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
    }
    return @intFromEnum(CError.Success);
}

export fn dve_rename(old_key: [*:0]const u8, new_key: [*:0]const u8) c_int {
    mutex.lock();
    defer mutex.unlock();

    if (!initialized) return @intFromEnum(CError.NotInit);
    const old_s = std.mem.sliceTo(old_key, 0);
    const new_s = std.mem.sliceTo(new_key, 0);
    switch (active_model) {
        .apple_nl => apple_db.?.renamePath(old_s, new_s) catch |err| {
            std.log.err("dve_rename: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
        .mpnet => mpnet_db.?.renamePath(old_s, new_s) catch |err| {
            std.log.err("dve_rename: {}\n", .{err});
            return @intFromEnum(CError.GenericFail);
        },
    }
    return @intFromEnum(CError.Success);
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
