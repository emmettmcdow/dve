// C bridge between llama.cpp and dve. See llama_bridge.h for the contract.
//
// Modelled on llama.cpp's tools/embedding, minus the batching: dve embeds one
// chunk at a time, so a single sequence per decode keeps this short.
#include "llama_bridge.h"

#include "llama.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Overridable with -DDVE_LLAMA_MODEL_PATH="..." at compile time, and by the
// DVE_LLAMA_MODEL environment variable at run time.
#ifndef DVE_LLAMA_MODEL_PATH
#define DVE_LLAMA_MODEL_PATH "/Users/emcdow/llama.cpp/build/bin/nomic-embed-text-v1.5.f32.gguf"
#endif

// nomic-embed-text-v1.5 trains at 2048 and stretches to 8192 with YaRN at
// freq_scale 0.75 -- the same knobs llama-embedding takes on the command line.
#define DVE_N_CTX          8192
#define DVE_ROPE_FREQ_SCALE 0.75f

// llama_context is not thread-safe, and neither is lazy init. One lock covers
// both; embedding is compute-bound inside llama.cpp anyway.
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static struct llama_model   *g_model;
static struct llama_context *g_ctx;
static int                   g_n_embd;
static int                   g_load_failed; // sticky: don't re-read a 500MB gguf per call

static void discard_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

// ggml-metal asserts at teardown that every Metal resource set has been
// released, so the context has to go before libggml's own static destructors
// run. Registering from ensure_loaded() puts this ahead of them in the
// atexit/__cxa_finalize order, since libggml was loaded first.
static void shutdown_bridge(void) {
    if (g_ctx != NULL) {
        llama_free(g_ctx);
        g_ctx = NULL;
    }
    if (g_model != NULL) {
        llama_model_free(g_model);
        g_model = NULL;
    }
    llama_backend_free();
}

// Caller must hold g_lock. Returns 0 on success.
static int ensure_loaded(void) {
    if (g_ctx != NULL) {
        return 0;
    }
    if (g_load_failed) {
        return -1;
    }

    if (getenv("DVE_LLAMA_VERBOSE") == NULL) {
        llama_log_set(discard_log, NULL);
    }
    llama_backend_init();

    const char *model_path = getenv("DVE_LLAMA_MODEL");
    if (model_path == NULL || model_path[0] == '\0') {
        model_path = DVE_LLAMA_MODEL_PATH;
    }

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    g_model = llama_model_load_from_file(model_path, mparams);
    if (g_model == NULL) {
        fprintf(stderr, "dve_embed: failed to load model '%s'\n", model_path);
        g_load_failed = 1;
        return -1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.embeddings        = true;
    cparams.pooling_type      = LLAMA_POOLING_TYPE_MEAN;
    cparams.n_ctx             = DVE_N_CTX;
    cparams.n_batch           = DVE_N_CTX; // must hold a whole sequence: no pooling across decodes
    cparams.n_ubatch          = DVE_N_CTX;
    cparams.n_seq_max         = 1;
    cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
    cparams.rope_freq_scale   = DVE_ROPE_FREQ_SCALE;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (g_ctx == NULL) {
        fprintf(stderr, "dve_embed: failed to create context\n");
        llama_model_free(g_model);
        g_model = NULL;
        g_load_failed = 1;
        return -1;
    }

    g_n_embd = llama_model_n_embd(g_model);
    atexit(shutdown_bridge);
    return 0;
}

int dve_embed(float *out, size_t out_len, const char *text) {
    if (out == NULL || text == NULL) {
        return DVE_EMBED_ERR_ARGS;
    }

    const size_t text_len = strlen(text);
    if (text_len == 0) {
        return DVE_EMBED_ERR_TOKENIZE;
    }

    pthread_mutex_lock(&g_lock);

    int rc;
    llama_token *tokens = NULL;
    struct llama_batch batch = {0};
    int batch_alloced = 0;

    if (ensure_loaded() != 0) {
        rc = DVE_EMBED_ERR_INIT;
        goto done;
    }
    if (out_len < (size_t)g_n_embd) {
        fprintf(stderr, "dve_embed: buffer holds %zu floats, model needs %d\n", out_len, g_n_embd);
        rc = DVE_EMBED_ERR_BUFFER;
        goto done;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);

    // A NULL destination makes llama_tokenize report the count it would need.
    const int32_t n_needed = -llama_tokenize(vocab, text, (int32_t)text_len, NULL, 0, true, true);
    if (n_needed <= 0) {
        rc = DVE_EMBED_ERR_TOKENIZE;
        goto done;
    }

    tokens = malloc((size_t)n_needed * sizeof(llama_token));
    if (tokens == NULL) {
        rc = DVE_EMBED_ERR_TOKENIZE;
        goto done;
    }

    int32_t n_tokens = llama_tokenize(vocab, text, (int32_t)text_len, tokens, n_needed, true, true);
    if (n_tokens <= 0) {
        rc = DVE_EMBED_ERR_TOKENIZE;
        goto done;
    }
    if (n_tokens > DVE_N_CTX) {
        n_tokens = DVE_N_CTX; // truncate rather than fail; callers chunk upstream
    }

    batch = llama_batch_init(n_tokens, 0, 1);
    batch_alloced = 1;
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1; // pooling needs every token marked as an output
    }

    // Embeddings carry no state between calls, so the KV cache from the last
    // one is not just useless but wrong.
    llama_memory_clear(llama_get_memory(g_ctx), true);

    if (llama_decode(g_ctx, batch) < 0) {
        rc = DVE_EMBED_ERR_DECODE;
        goto done;
    }

    const float *embd = llama_get_embeddings_seq(g_ctx, 0);
    if (embd == NULL) {
        rc = DVE_EMBED_ERR_NO_EMBD;
        goto done;
    }

    // L2 normalize so callers can use a dot product for cosine similarity.
    double sum = 0.0;
    for (int i = 0; i < g_n_embd; i++) {
        sum += (double)embd[i] * (double)embd[i];
    }
    const float norm = sum > 0.0 ? (float)(1.0 / sqrt(sum)) : 0.0f;
    for (int i = 0; i < g_n_embd; i++) {
        out[i] = embd[i] * norm;
    }

    rc = g_n_embd;

done:
    if (batch_alloced) {
        llama_batch_free(batch);
    }
    free(tokens);
    pthread_mutex_unlock(&g_lock);
    return rc;
}
