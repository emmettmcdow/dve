// C bridge between llama.cpp and dve.
//
// The whole point of this file is to keep llama.cpp's API on the C side of the
// fence: Zig only ever sees `dve_embed`, three plain arguments and an int.
#ifndef DVE_LLAMA_BRIDGE_H
#define DVE_LLAMA_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Negative return values from dve_embed.
#define DVE_EMBED_ERR_ARGS     (-1) // out/text was NULL
#define DVE_EMBED_ERR_INIT     (-2) // model or context failed to load
#define DVE_EMBED_ERR_BUFFER   (-3) // out_len < the model's embedding dimension
#define DVE_EMBED_ERR_TOKENIZE (-4) // text tokenized to nothing, or tokenizing failed
#define DVE_EMBED_ERR_DECODE   (-5) // llama_decode failed
#define DVE_EMBED_ERR_NO_EMBD  (-6) // decode succeeded but no sequence embedding came back

// Embeds `text` (NUL-terminated, UTF-8) into `out`, which must have room for at
// least the model's embedding dimension (768 for nomic-embed-text-v1.5).
//
// The result is L2-normalized, so cosine similarity is a plain dot product.
//
// Returns the number of floats written, or one of the DVE_EMBED_ERR_* codes
// above. Thread-safe: calls are serialized internally.
//
// The model is loaded lazily on the first call and kept for the lifetime of the
// process. Its path comes from the DVE_LLAMA_MODEL environment variable, or the
// DVE_LLAMA_MODEL_PATH compile-time default. llama.cpp's own logging is
// suppressed unless DVE_LLAMA_VERBOSE is set in the environment.
int dve_embed(float *out, size_t out_len, const char *text);

// Embeds `n_texts` texts in one go, packing several sequences into each
// llama_decode. `outs[i]` receives text i's embedding and must have room for
// at least the model's embedding dimension, which `out_len` states once for
// all of them; results are L2-normalized exactly as dve_embed's are.
//
// Returns the number of floats written per text, or one of the DVE_EMBED_ERR_*
// codes above. An error abandons the whole call: some of `outs` may have been
// written, and none of it should be used. Embedding N texts this way gives the
// same vectors as N dve_embed calls -- dve_embed is itself a batch of one --
// but is several times faster for short texts, where a one-sequence decode
// spends most of its time on dispatch overhead rather than on the model.
//
// Thread-safe on the same terms as dve_embed: calls are serialized internally.
int dve_embed_batch(float *const *outs, size_t out_len, const char *const *texts, size_t n_texts);

#ifdef __cplusplus
}
#endif

#endif // DVE_LLAMA_BRIDGE_H
