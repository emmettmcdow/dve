# Embedding throughput: llama.cpp vs CoreML, and what batching is worth

Run 2026-09-25 on an Apple M5 (4 performance + 6 efficiency cores, Metal),
`-Doptimize=ReleaseFast -Dllama`. Corpus is 300 Simple Wikipedia articles
sampled with `--seed 1`: **36,597 sentences**, 122.0 per article, 35.9 bytes
each (~13.4 tokens). Harness is `experiments/embedbench/`.

Raw output in `raw/embedbench-20260925-1735.txt`,
`raw/embedbench-order-*.txt`, `raw/llama-embedding-seqsweep-*.txt`.

## The question

Two changes landed close together and are easy to conflate:

- `7003133` added the llama.cpp / nomic-embed-text-v1.5 backend.
- `c2c0d4b` + `62748af` added `embedBatch` and moved `embedTextInternal` onto it.

Only the first one made anything faster. The second is plumbing.

## 1. Batching is currently a no-op, by construction

All three backends register `.embedBatchFn = sequentialEmbedBatch(embed)`, which
is a `for` loop over the single-string `embed`. `c2c0d4b`'s commit message says
so outright: *"Every backend still batches by looping over embed, so behavior is
unchanged until one gets a native batch."*

Measured, 36,597 sentences, one `embedBatch` per document against one `embed`
per sentence:

| model | single (sent/s) | batch (sent/s) | batch/single |
|---|---:|---:|---:|
| llama | 87.1 | 90.8 | 1.043x |
| mpnet | 46.3 | 49.8 | 1.075x |

The 4-7% is not batching. `single` was timed first in both runs and paid for
warming the machine. Re-running at 100 articles in both orders:

| model | order | single | batch | batch/single |
|---|---|---:|---:|---:|
| llama | single first | 91.9 | 92.7 | 1.009x |
| llama | batch first  | 92.7 | 92.7 | 1.000x |
| mpnet | single first | 50.2 | 50.2 | 1.002x |
| mpnet | batch first  | 50.3 | 50.1 | 1.003x |

Flip the order and the difference vanishes. **Batching buys 1.00x today.**
That is the expected answer: the call sequence reaching the model is identical.
`--reverse` exists in the harness because of this.

## 2. llama.cpp is ~1.9x CoreML mpnet

Same sentences, same machine:

| backend | sent/s | ms/sentence |
|---|---:|---:|
| llama nomic-embed-text-v1.5 (f32, Metal) | 92.7 | 10.8 |
| mpnet (CoreML) | 50.2 | 19.9 |

**1.85x**, and llama's vectors are 768-d like mpnet's, so nothing downstream
changes. Model load is the one regression: 12.2 s cold for the 547 MB f32 gguf
against 1.0 s for the CoreML bundle. Warm (page cache hot) it is 0.8 s.

## 3. The bridge leaves ~5x on the table

`src/llama_bridge.c` sets `n_seq_max = 1` and does one `llama_tokenize` pair,
one `llama_batch_init`/`llama_batch_free`, one `llama_memory_clear` and one
`llama_decode` **per sentence**. At 92.7 sent/s and 13.4 tokens per sentence
that is ~1,250 tokens/sec out of a 137M-parameter model with all layers offloaded
to Metal -- nearly all of it dispatch overhead against an `n_ubatch` of 8192 that
is being handed twelve tokens.

llama.cpp's own `llama-embedding`, on exactly the sentences dve embeds
(`embedbench --dump`), same model and same machine, packing N sequences into one
`llama_decode`:

| sequences per decode | sent/s | vs dve's bridge |
|---:|---:|---:|
| 1 (what dve does) | 92.7 | 1.0x |
| 16 | 224 | 2.4x |
| 32 | 306 | 3.3x |
| 64 | 403 | 4.3x |
| 128 | 543 | 5.9x |
| 250 | 485-522 | 5.2-5.6x |

It saturates around 128 sequences per decode at roughly **5x**, or ~6,500
tokens/sec. The points at 128 and above differ by less than the noise in
subtracting model-load time from a sub-second decode; treat the plateau as ~5x,
not as a real peak at 128.

**The knee sits exactly where our documents do.** Articles average 122 sentences,
so a native batch that embeds one document per `llama_decode` -- which is already
the call shape `62748af` set up -- should reach the plateau without batching
across documents.

## What this costs at corpus scale

Simple Wikipedia is ~36.9M chunks (see `src/chunking.md`):

| backend | full-corpus embed time |
|---|---:|
| mpnet (CoreML) | ~205 h |
| llama, today | ~110 h |
| llama, native batch at 5x | ~22 h |

## 4. Native batching, built [2026-09-25]

`dve_embed_batch` packs up to `DVE_MAX_SEQ` (128) sequences into one
`llama_decode`, cutting a batch when either that limit or the 8192-token budget
is reached. `dve_embed` is now a batch of one through the same path, so the two
cannot drift. `n_seq_max` went from 1 to 128 and `n_ctx` from 8192 to
128 * 8192, because llama.cpp divides the context evenly among sequences and the
per-sequence budget had to stay at 8192.

Measured, same corpus and machine:

| | sent/s | ms/sentence |
|---|---:|---:|
| before (1 sequence per decode) | 87.1 | 11.49 |
| after, 100 articles | 711-713 | 1.40 |
| after, 300 articles | **772.0** | **1.30** |

**8.9x**, and it does not sag with scale -- 300 articles came out faster than
100, as more documents amortize more. Confirmed against ordering: reversing the
two modes gives 8.11x against 8.03x.

That beats the ~5x `llama-embedding` suggested. Its numbers carried its own
per-run overhead and a model-load subtraction that was only good to a few tens
of milliseconds; treat it as the lower bound it was.

### Batching is not bit-exact, and cannot be

Packing several sequences into a decode changes the shape of the matmuls, so
the GPU's reductions come out in a different order. Over all 36,597 sentences
(`embedbench --verify`):

```
compared        36597 vectors
null mismatches 0
max |delta|     7.251e-4
min cosine      0.999982600
```

A cosine of 0.99998 is reduction noise. Sequences leaking into each other
through a shared decode -- the failure this design could plausibly have -- would
show a cosine far below 1, so this is the check worth keeping. The unit test
that asserted bitwise equality was asserting a guarantee llama.cpp never made;
it now takes a tolerance, which is 0 for backends whose batch is still
`sequentialEmbedBatch` and 2e-3 for llama.

Two tests were added for the packing loop specifically: 300 distinct strings, to
span several decodes and catch a vector misfiled by one, and 40 long strings, to
cut a batch on the token budget rather than the sequence count.

### What it costs

Peak RSS goes from 0.59 GB to 0.93 GB. The larger `n_ctx` is not what costs it
-- single-sequence work under the new context still peaks at 0.59 GB -- it is
the transient compute buffer for decoding ~122 sequences at once, and it is
bounded by `DVE_MAX_SEQ` and `DVE_N_BATCH`.

## Corpus scale, revised

Simple Wikipedia, ~36.9M chunks:

| backend | full-corpus embed time |
|---|---:|
| mpnet (CoreML) | ~205 h |
| llama, one sequence per decode | ~110 h |
| llama, native batch | **~13 h** |

## Conclusions

1. `embedBatch` made nothing faster and was never going to; it made a native
   batch possible. `62748af` was the setup, not the win.
2. Switching to llama.cpp is worth 1.85x on its own.
3. Native batching is worth a further **8.9x**, for 15.4x over CoreML mpnet
   end to end. It cost ~340 MB of peak RSS and no accuracy that a cosine of
   0.99998 can detect.
4. The remaining backends still use `sequentialEmbedBatch`. CoreML supports
   batched prediction, so mpnet has a similar win available; nobody has tried it.
5. Do not benchmark embedding on a laptop running on battery in a bag. The
   300-article run came in 4-6% under the 100-article run purely on heat.
