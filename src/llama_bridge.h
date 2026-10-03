// C bridge between llama.cpp and dve.
//
// The whole point of this file is to keep llama.cpp's API on the C side of the
// fence: Zig only ever sees plain arguments and an int.
#ifndef DVE_LLAMA_BRIDGE_H
#define DVE_LLAMA_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Negative return values from dve_embed.
#define DVE_EMBED_ERR_ARGS     (-1) // out/text was NULL
#define DVE_EMBED_ERR_INIT     (-2) // model or context failed to load, or was never loaded
#define DVE_EMBED_ERR_BUFFER   (-3) // out_len < the model's embedding dimension
#define DVE_EMBED_ERR_TOKENIZE (-4) // text tokenized to nothing, or tokenizing failed
#define DVE_EMBED_ERR_DECODE   (-5) // llama_decode failed
#define DVE_EMBED_ERR_NO_EMBD  (-6) // decode succeeded but no sequence embedding came back
#define DVE_EMBED_ERR_LOADED   (-7) // a model from a different path is already loaded

// Loads the .gguf at `model_path` (NUL-terminated) and keeps it for the
// lifetime of the process. Must succeed before dve_embed or dve_embed_batch is
// called. Loading the same path again is a no-op; the bridge holds one model,
// so a different path once one is loaded is DVE_EMBED_ERR_LOADED. A failed load
// leaves nothing behind and may be retried.
//
// Returns 0 on success, or one of the DVE_EMBED_ERR_* codes above. Thread-safe
// on the same terms as dve_embed. llama.cpp's own logging is suppressed unless
// DVE_LLAMA_VERBOSE is set in the environment.
int dve_embed_load(const char *model_path);

// Embeds `text` (NUL-terminated, UTF-8) into `out`, which must have room for at
// least the model's embedding dimension (768 for nomic-embed-text-v1.5).
//
// The result is L2-normalized, so cosine similarity is a plain dot product.
//
// Returns the number of floats written, or one of the DVE_EMBED_ERR_* codes
// above, DVE_EMBED_ERR_INIT if dve_embed_load has not succeeded. Thread-safe:
// calls are serialized internally.
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
