# wikitest: the engine on a real corpus

Apple M5, 24 GB RAM, zig 0.15.2, `-Doptimize=ReleaseFast`. 2,000 Simple Wikipedia articles
sampled with `--sample 1`, embedded with mpnet to **247,396 vectors** of 768 f32. Run as:

```sh
(cd wikitest && ../zig-out/bin/wikitest embed --limit 2000 --sample 1 --db <dir>)
(cd wikitest && ../zig-out/bin/wikitest search --db <dir> -k 10 --repeat 7 --verify)
```

This is the first measurement of the whole stack -- `vstore.zig` for bytes, `codes.zig` for
stage one, `vector.zig` wiring them -- on embeddings a real model produced from real text, in
answer to queries a person would type. Everything before it was synthetic vectors, a 37k
corpus, or both.

## Recall holds at 0.987, and the queries are the point

15 natural-language queries, top-10, measured against an exhaustive scan of the same store:

| | |
|---|---:|
| **recall** | **0.987** (13 queries at 1.000, two at 0.900) |
| search | 1.4-2.4 ms typical, 4.6 ms worst median seen |
| exhaustive scan | 117.4 ms average |
| **speedup** | **~75x** |

**This closes the caveat `binrecall.md` was written with.** Its queries were corpus vectors --
standard ANN-benchmark practice, but not what a person types, and a short question's embedding
could plausibly sit further from every document than documents sit from each other. It does
not. 0.987 here against 0.996 predicted there, on 6.6x the corpus, with real queries.

The two queries at 0.900 are ties: their exhaustive results contained equal similarities, so
which ten come back is arbitrary among equals. Same effect `binrecall.md` documents at length.

## The bottleneck is the embedder now, by an order of magnitude

| | |
|---|---:|
| embed a query (CoreML, warm) | 10-27 ms |
| search | **1.4-2.4 ms** |

Storage and indexing are no longer what makes a query slow. That is worth saying plainly
because it is the first time it has been true, and it means further work on the scan buys
nothing a user would feel until the embedder moves.

Ingest is the same story from the other side: **247,396 vectors took 1h46m**, about 39 vectors
a second, all of it CoreML. Projected to the ~35M vectors of the full corpus that is roughly
ten days. wikitest can validate the design at a few hundred thousand vectors; every number in
this repo for 35M remains a projection.

## Startup, and what the codes file is for

Measured by differencing a 247k database against an empty one, three runs each, all stable to
a few milliseconds:

| | |
|---|---:|
| CoreML model load | 1.018 s |
| store scan + code build (247k vectors, 966 MB) | **0.279 s** |
| total | 1.297 s |

279 ms is fine. It does not stay fine: the rebuild reads **every vector**, and at 35M the store
is ~143 GB, which no page cache holds. At the 6.6 GiB/s this machine does cold sequential that
is **~22 seconds minimum** at every launch, against ~0.3 s to read a 1.68 GB codes file. That
is the whole argument for persisting the codes, and it is now a measured argument rather than
an assumed one.

Peak RSS is 976.6 MB, essentially all of it the embedding model. The codes for 247k vectors are
11.9 MB.

## The store survived a crash, by accident

Partway through the ingest this file's author concluded the embedder had died and opened the
database mid-write to see what was left. It came back **145,758 live / 145,758 slots** -- no
holes, no corruption, no recovery step. The process had not in fact died, so this was also an
unplanned test of a concurrent reader against an active writer, which it passed.

Neither was intentional and neither is a substitute for a real crash test, but page-aligned
chunks with metadata trailers and a crc did what `vec_storage2.md` says they do.

## What this does not measure

- **Cold.** The store was written minutes earlier and the queries ran warm. Stage two's
  candidate reads are the part that would change, and by the most.
- **Scale.** 247k vectors against a ~35M target, and the embedder is why.
- **Result quality.** Recall is agreement with an exhaustive scan, not relevance. A random
  2,000-article sample often has no good answer to a given question, and the top hits show it.
