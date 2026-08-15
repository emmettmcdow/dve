//! Zig side of the llama.cpp C bridge.
//!
//! All of llama.cpp's API lives in src/llama_bridge.c; this file only declares
//! the one symbol that crosses over. Build wiring is gated on the embedding
//! model, so llama.cpp is neither compiled nor linked unless it is selected.

pub const enabled = config.embedding_model == .llama_nomic_embed_text_v1_5_f32;

pub const Error = error{
    /// Built without -Dembedding-model=llama_nomic_embed_text_v1_5_f32, so the
    /// bridge was never linked in.
    LlamaNotLinked,
    BadArgs,
    InitFailed,
    BufferTooSmall,
    TokenizeFailed,
    DecodeFailed,
    NoEmbedding,
};

/// Embeds `text` into `out`, returning the number of floats written. `out` must
/// have room for the model's embedding dimension (768). The result is
/// L2-normalized.
pub fn embed(out: []f32, text: [:0]const u8) Error!usize {
    if (comptime !enabled) return Error.LlamaNotLinked;

    const rc = dve_embed(out.ptr, out.len, text.ptr);
    if (rc >= 0) return @intCast(rc);
    return switch (rc) {
        -1 => Error.BadArgs,
        -2 => Error.InitFailed,
        -3 => Error.BufferTooSmall,
        -4 => Error.TokenizeFailed,
        -5 => Error.DecodeFailed,
        -6 => Error.NoEmbedding,
        else => Error.InitFailed,
    };
}

extern fn dve_embed(out: [*]f32, out_len: usize, text: [*:0]const u8) c_int;

const config = @import("config");
