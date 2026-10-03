//! Zig side of the llama.cpp C bridge.
//!
//! All of llama.cpp's API lives in src/llama_bridge.c; this file only declares
//! the symbols that cross over. Build wiring is gated on -Dllama, so
//! llama.cpp is neither compiled nor linked unless that flag is set.

pub const enabled = config.llama;

pub const Error = error{
    /// Built without -Dllama, so the bridge was never linked in.
    LlamaNotLinked,
    BadArgs,
    InitFailed,
    BufferTooSmall,
    TokenizeFailed,
    DecodeFailed,
    NoEmbedding,
    /// The bridge holds one model per process, and it came from another path.
    ModelAlreadyLoaded,
};

/// Loads the .gguf at `model_path`, which every `embed` after it then uses.
/// Loading the same path again is a no-op.
pub fn load(model_path: [:0]const u8) Error!void {
    if (comptime !enabled) return Error.LlamaNotLinked;

    const rc = dve_embed_load(model_path.ptr);
    if (rc != 0) return errorFor(rc);
}

/// Embeds `text` into `out`, returning the number of floats written. `out` must
/// have room for the model's embedding dimension (768). The result is
/// L2-normalized.
pub fn embed(out: []f32, text: [:0]const u8) Error!usize {
    if (comptime !enabled) return Error.LlamaNotLinked;

    const rc = dve_embed(out.ptr, out.len, text.ptr);
    if (rc >= 0) return @intCast(rc);
    return errorFor(rc);
}

/// Embeds every text in `texts` in one call, writing text i's vector to
/// `outs[i]`, which must have room for the model's embedding dimension (768).
/// Results are L2-normalized and identical to what `embed` would produce for
/// each text alone; batching only changes how many sequences share a decode.
///
/// An error abandons the whole call -- some of `outs` may have been written.
pub fn embedBatch(outs: []const [*]f32, out_len: usize, texts: []const [*:0]const u8) Error!usize {
    if (comptime !enabled) return Error.LlamaNotLinked;
    assert(outs.len == texts.len);
    if (texts.len == 0) return 0;

    const rc = dve_embed_batch(outs.ptr, out_len, texts.ptr, texts.len);
    if (rc >= 0) return @intCast(rc);
    return errorFor(rc);
}

fn errorFor(rc: c_int) Error {
    return switch (rc) {
        -1 => Error.BadArgs,
        -2 => Error.InitFailed,
        -3 => Error.BufferTooSmall,
        -4 => Error.TokenizeFailed,
        -5 => Error.DecodeFailed,
        -6 => Error.NoEmbedding,
        -7 => Error.ModelAlreadyLoaded,
        else => Error.InitFailed,
    };
}

extern fn dve_embed_load(model_path: [*:0]const u8) c_int;
extern fn dve_embed(out: [*]f32, out_len: usize, text: [*:0]const u8) c_int;
extern fn dve_embed_batch(
    outs: [*]const [*]f32,
    out_len: usize,
    texts: [*]const [*:0]const u8,
    n_texts: usize,
) c_int;

const std = @import("std");
const assert = std.debug.assert;
const config = @import("config");
