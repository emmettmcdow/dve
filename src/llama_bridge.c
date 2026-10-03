// C bridge between llama.cpp and dve. See llama_bridge.h for the contract.
//
// Modelled on llama.cpp's tools/embedding. One llama_decode carries up to
// DVE_MAX_SEQ sequences: a decode costs about the same whether it is handed
// twelve tokens or a few thousand, so embedding one short sentence at a time
// spends nearly all its time on dispatch overhead.
#include "llama_bridge.h"

#include "llama.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// nomic-embed-text-v1.5 trains at 2048 and stretches to 8192 with YaRN at
// freq_scale 0.75 -- the same knobs llama-embedding takes on the command line.
#define DVE_N_CTX_PER_SEQ   8192
#define DVE_ROPE_FREQ_SCALE 0.75f

// Sequences packed into one llama_decode. Throughput climbs steeply to about
// 64 and is flat past ~128 (experiments/results/embedbench.md), and documents
// average 122 chunks, so 128 reaches the plateau on one document's worth of
// text without batching across documents.
#define DVE_MAX_SEQ 128

// Tokens per decode. Equal to the per-sequence context so that one maximal
// sequence still fits in a batch by itself, which is what lets the packing
// loop below assume it can always make progress.
#define DVE_N_BATCH DVE_N_CTX_PER_SEQ

// llama.cpp divides the context evenly among sequences, so the total has to be
// scaled up to keep each one's budget at DVE_N_CTX_PER_SEQ. The declaration
// itself is free -- an encoder has no KV cache to grow, and single-sequence
// work under this context peaks where it did before. What batching does cost
// is the compute buffer for a full decode: ~340MB of peak RSS, against 547MB
// of model.
#define DVE_N_CTX (DVE_N_CTX_PER_SEQ * DVE_MAX_SEQ)

// llama_context is not thread-safe, and neither is loading. One lock covers
// both; embedding is compute-bound inside llama.cpp anyway.
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static struct llama_model   *g_model;
static struct llama_context *g_ctx;
static int                   g_n_embd;
static char                 *g_model_path; // what g_model was loaded from
static int                   g_backend_up;
static int                   g_atexit_set;

static void discard_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

// ggml-metal asserts at teardown that every Metal resource set has been
// released, so the context has to go before libggml's own static destructors
// run. Registering once a context exists puts this ahead of them in the
// atexit/__cxa_finalize order: by then ggml-metal has registered its own.
static void shutdown_bridge(void) {
    if (g_ctx != NULL) {
        llama_free(g_ctx);
        g_ctx = NULL;
    }
    if (g_model != NULL) {
        llama_model_free(g_model);
        g_model = NULL;
    }
    free(g_model_path);
    g_model_path = NULL;
    llama_backend_free();
}

// Caller must hold g_lock.
static int load_locked(const char *model_path) {
    if (g_ctx != NULL) {
        return strcmp(model_path, g_model_path) == 0 ? 0 : DVE_EMBED_ERR_LOADED;
    }

    char *path_copy = strdup(model_path);
    if (path_copy == NULL) {
        return DVE_EMBED_ERR_INIT;
    }

    if (!g_backend_up) {
        if (getenv("DVE_LLAMA_VERBOSE") == NULL) {
            llama_log_set(discard_log, NULL);
        }
        llama_backend_init();
        g_backend_up = 1;
    }

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    g_model = llama_model_load_from_file(model_path, mparams);
    if (g_model == NULL) {
        fprintf(stderr, "dve_embed: failed to load model '%s'\n", model_path);
        free(path_copy);
        return DVE_EMBED_ERR_INIT;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.embeddings        = true;
    cparams.pooling_type      = LLAMA_POOLING_TYPE_MEAN;
    cparams.n_ctx             = DVE_N_CTX;
    cparams.n_batch           = DVE_N_BATCH; // must hold a whole sequence: no pooling across decodes
    cparams.n_ubatch          = DVE_N_BATCH;
    cparams.n_seq_max         = DVE_MAX_SEQ;
    cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
    cparams.rope_freq_scale   = DVE_ROPE_FREQ_SCALE;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (g_ctx == NULL) {
        fprintf(stderr, "dve_embed: failed to create context\n");
        llama_model_free(g_model);
        g_model = NULL;
        free(path_copy);
        return DVE_EMBED_ERR_INIT;
    }

    g_n_embd = llama_model_n_embd(g_model);
    g_model_path = path_copy;
    if (!g_atexit_set) {
        atexit(shutdown_bridge);
        g_atexit_set = 1;
    }
    return 0;
}

int dve_embed_load(const char *model_path) {
    if (model_path == NULL) {
        return DVE_EMBED_ERR_ARGS;
    }
    pthread_mutex_lock(&g_lock);
    const int rc = load_locked(model_path);
    pthread_mutex_unlock(&g_lock);
    return rc;
}

// Tokenizes `text` into a freshly malloc'd array, which the caller frees.
// Returns the token count, or a negative DVE_EMBED_ERR_* code; *out_tokens is
// written only on success.
static int tokenize_one(const struct llama_vocab *vocab, const char *text,
                        llama_token **out_tokens) {
    const size_t text_len = strlen(text);
    if (text_len == 0) {
        return DVE_EMBED_ERR_TOKENIZE;
    }

    // A NULL destination makes llama_tokenize report the count it would need.
    const int32_t n_needed = -llama_tokenize(vocab, text, (int32_t)text_len, NULL, 0, true, true);
    if (n_needed <= 0) {
        return DVE_EMBED_ERR_TOKENIZE;
    }

    llama_token *tokens = malloc((size_t)n_needed * sizeof(llama_token));
    if (tokens == NULL) {
        return DVE_EMBED_ERR_TOKENIZE;
    }

    int32_t n_tokens = llama_tokenize(vocab, text, (int32_t)text_len, tokens, n_needed, true, true);
    if (n_tokens <= 0) {
        free(tokens);
        return DVE_EMBED_ERR_TOKENIZE;
    }
    if (n_tokens > DVE_N_CTX_PER_SEQ) {
        n_tokens = DVE_N_CTX_PER_SEQ; // truncate rather than fail; callers chunk upstream
    }

    *out_tokens = tokens;
    return n_tokens;
}

// Copies sequence `seq`'s pooled embedding out of the context into `out`,
// L2 normalized so callers can use a dot product for cosine similarity.
// Caller must hold g_lock.
static int read_embedding(int32_t seq, float *out) {
    const float *embd = llama_get_embeddings_seq(g_ctx, seq);
    if (embd == NULL) {
        return DVE_EMBED_ERR_NO_EMBD;
    }

    double sum = 0.0;
    for (int i = 0; i < g_n_embd; i++) {
        sum += (double)embd[i] * (double)embd[i];
    }
    const float norm = sum > 0.0 ? (float)(1.0 / sqrt(sum)) : 0.0f;
    for (int i = 0; i < g_n_embd; i++) {
        out[i] = embd[i] * norm;
    }
    return g_n_embd;
}

int dve_embed_batch(float *const *outs, size_t out_len, const char *const *texts, size_t n_texts) {
    if (outs == NULL || texts == NULL) {
        return DVE_EMBED_ERR_ARGS;
    }

    pthread_mutex_lock(&g_lock);

    int rc;
    llama_token **toks = NULL;
    int32_t *tok_n = NULL;
    struct llama_batch batch = {0};
    int batch_alloced = 0;

    if (g_ctx == NULL) {
        rc = DVE_EMBED_ERR_INIT;
        goto done;
    }
    if (n_texts == 0) {
        rc = g_n_embd;
        goto done;
    }
    if (out_len < (size_t)g_n_embd) {
        fprintf(stderr, "dve_embed: buffer holds %zu floats, model needs %d\n", out_len, g_n_embd);
        rc = DVE_EMBED_ERR_BUFFER;
        goto done;
    }

    toks = calloc(n_texts, sizeof(*toks));
    tok_n = calloc(n_texts, sizeof(*tok_n));
    if (toks == NULL || tok_n == NULL) {
        rc = DVE_EMBED_ERR_TOKENIZE;
        goto done;
    }

    // Tokenize everything up front: the packing loop needs each length to know
    // where to cut a batch, and a failure here should not leave half the
    // outputs written.
    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);
    for (size_t i = 0; i < n_texts; i++) {
        if (texts[i] == NULL || outs[i] == NULL) {
            rc = DVE_EMBED_ERR_ARGS;
            goto done;
        }
        const int n = tokenize_one(vocab, texts[i], &toks[i]);
        if (n < 0) {
            rc = n;
            goto done;
        }
        tok_n[i] = (int32_t)n;
    }

    batch = llama_batch_init(DVE_N_BATCH, 0, DVE_MAX_SEQ);
    batch_alloced = 1;

    size_t i = 0;
    while (i < n_texts) {
        // Fill the batch until either limit would be exceeded. Progress is
        // guaranteed: tokenize_one truncates at DVE_N_CTX_PER_SEQ, which is
        // DVE_N_BATCH, so the first sequence always fits on its own.
        int32_t n_tokens = 0;
        int32_t n_seq = 0;
        size_t j = i;
        while (j < n_texts && n_seq < DVE_MAX_SEQ && n_tokens + tok_n[j] <= DVE_N_BATCH) {
            for (int32_t t = 0; t < tok_n[j]; t++) {
                batch.token[n_tokens]     = toks[j][t];
                batch.pos[n_tokens]       = t;
                batch.n_seq_id[n_tokens]  = 1;
                batch.seq_id[n_tokens][0] = n_seq;
                batch.logits[n_tokens]    = 1; // pooling needs every token marked as an output
                n_tokens++;
            }
            n_seq++;
            j++;
        }
        if (n_seq == 0) {
            rc = DVE_EMBED_ERR_DECODE; // unreachable unless the invariant above breaks
            goto done;
        }
        batch.n_tokens = n_tokens;

        // Embeddings carry no state between decodes, so the previous batch's
        // cache is not just useless but wrong.
        llama_memory_clear(llama_get_memory(g_ctx), true);

        if (llama_decode(g_ctx, batch) < 0) {
            rc = DVE_EMBED_ERR_DECODE;
            goto done;
        }

        for (int32_t s = 0; s < n_seq; s++) {
            const int r = read_embedding(s, outs[i + (size_t)s]);
            if (r < 0) {
                rc = r;
                goto done;
            }
        }
        i = j;
    }

    rc = g_n_embd;

done:
    if (batch_alloced) {
        llama_batch_free(batch);
    }
    if (toks != NULL) {
        for (size_t k = 0; k < n_texts; k++) {
            free(toks[k]);
        }
        free(toks);
    }
    free(tok_n);
    pthread_mutex_unlock(&g_lock);
    return rc;
}

// One text is just a batch of one. Sharing the path is what keeps dve_embed and
// dve_embed_batch from drifting apart, which the Zig side asserts by checking
// that a batch equals the same strings embedded one at a time.
int dve_embed(float *out, size_t out_len, const char *text) {
    if (out == NULL || text == NULL) {
        return DVE_EMBED_ERR_ARGS;
    }
    return dve_embed_batch(&out, out_len, &text, 1);
}
