#pragma once
#include <stdint.h>

#define DVE_PATH_MAX 1024

typedef enum {
    DVE_SUCCESS        =  0,
    DVE_ERR_GENERIC    = -1,
    DVE_ERR_DOUBLE_INIT = -2,
    DVE_ERR_NOT_INIT   = -3,
} DveError;

typedef struct {
    char key[DVE_PATH_MAX];
    uint32_t start_i;
    uint32_t end_i;
    float similarity;
} DveSearchResult;

/**
 * Initialize the dve database.
 *
 * basedir        - absolute path to the directory where the database will be stored.
 * model_path     - absolute path to the .mlpackage or .mlmodelc model file.
 *                  Pass NULL or empty string when compiled for apple_nlembedding.
 * tokenizer_path - absolute path to the tokenizer.json file.
 *                  Pass NULL or empty string when compiled for apple_nlembedding.
 *
 * Returns DVE_SUCCESS on success, negative DveError on failure.
 */
int dve_init(const char *basedir, const char *model_path, const char *tokenizer_path);

/** Deinitialize the database and free all resources. */
int dve_deinit(void);

/**
 * Embed text synchronously. Blocks until embedding is complete.
 * key     - identifier for this piece of text (e.g. a file path or arbitrary key).
 * content - the text to embed.
 */
int dve_embed(const char *key, const char *content);

/**
 * Embed text asynchronously on a background thread.
 * key     - identifier for this piece of text.
 * content - the text to embed.
 */
int dve_embed_async(const char *key, const char *content);

/**
 * Search the database for text semantically similar to query.
 * query  - the search query.
 * outbuf - caller-provided buffer to write results into.
 * n      - capacity of outbuf.
 *
 * Returns the number of results written, or negative DveError on failure.
 */
int dve_search(const char *query, DveSearchResult *outbuf, uint32_t n);

/** Remove all embeddings associated with key. */
int dve_remove(const char *key);

/** Rename a key, preserving its embeddings. */
int dve_rename(const char *old_key, const char *new_key);
