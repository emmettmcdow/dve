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

---

# 1.18M vectors, llama backend [2026-09-28]

Apple M5, `-Dllama -Doptimize=ReleaseFast`. 4.8x the corpus above, and a different embedder:
`wikitest --model llama` selects the llama.cpp / nomic-embed-text-v1.5 backend, which since
the native batch runs at 772 chunks/sec standalone.

Raw output in `raw/wikitest-llama-20260928-2013.txt`.

## Search holds up

| | |
|---|---|
| vectors | **1,178,132** over 9,476 articles |
| store | 4.5 GB on disk |
| codes | 56.7 MB |
| open | 416 ms with the index on disk |
| query embed | 12-16 ms |
| **search** | **2.2-6.9 ms** median, k=10 |
| exhaustive scan | 710 ms |
| **speedup** | **~250-300x** |
| peak RSS | **653 MB**, of which 547 MB is the llama model |
| recall@10 | 0.950 over 6 queries |

Search is still single-digit milliseconds at 4.8x the vectors, and the resident cost of the
index is 57 MB against a 4.5 GB store. Query embedding is again several times the search.

Recall is 0.950 here against 0.987 above, but over only 6 queries on a different model, and
one query carries it (`photosynthesis in plants` at 0.900 while three others are 1.000). Not
enough queries to call it a regression.

## Ingest is quadratic in corpus size

This is the finding. Per-document ingest cost grows linearly with the store, so total ingest
is O(n^2):

| articles | vectors | store | vec/s | s/doc |
|---:|---:|---:|---:|---:|
| 2,000 | 126,732 | 0.50 GB | 1,277 | 0.102 |
| 3,000 | 245,027 | 0.96 GB | 897 | 0.132 |
| 4,000 | 363,369 | 1.4 GB | 375 | 0.298 |
| 5,000 | 476,446 | 1.8 GB | 280 | 0.428 |
| 7,000 | 742,374 | 2.8 GB | 216 | 0.730 |
| 9,000 | 994,364 | 3.8 GB | 164 | 0.650 |

**7.8x slower per document** between 2,000 and 9,000 articles. The embedder is not the cause;
it is constant at 772 chunks/sec.

`replaceVectors` (vector.zig) calls `vec_storage.vecsForDoc` for every document, and
`vecsForDocLocked` (vstore.zig) walks every chunk in the store reading metadata trailers. Each
document therefore scans the whole database looking for rows to replace -- including documents
that have never been seen and cannot have any.

Subtracting embed time (vectors-per-doc / 772) from each interval leaves 0.15 s at 1.4 GB,
0.42 s at 2.3 GB and 0.49 s at 3.8 GB: **store_bytes / ~6 GB/s**, a page-cached full scan per
document. The arithmetic matches the mechanism.

At the full 36.9M-chunk corpus (~151 GB) that is ~25 s per document. Wikipedia would not
finish.

This predates the llama work; it has been in `replaceVectors` since the v2 cutover. Native
batching is what exposed it. At the old 11.5 ms/chunk, embedding cost ~1.4 s/doc and buried a
0.5 s scan; at 1.3 ms/chunk the scan is 75% of ingest.

The likely fix is cheap: `note_id_map.getId(path) == null` means the document is new, so there
are no old rows and the scan can be skipped entirely -- which covers all of bulk ingest. One
thing to check first: that scan currently also reaps rows whose `doc_id` was reused after a
lost `.dve_ids`, so skipping it changes behaviour in that already-broken state, where
`validate()` and `pruneOrphanedPaths` are the right answers.

## Two harness bugs this run found

**`--csv` corrupted the stack.** `Csv.open` built the struct in a local, pointed `writer` at
that local's `buf`, and returned it by value. The writer then addressed a dead stack frame, so
every row went into reclaimed memory; after 10,500 articles the run took a SIGSEGV inside
float formatting (`fmt.float.binaryToDecimal`). The empty CSV and the crash were the same bug.
`Csv.init` now initializes in place.

The store came through it clean: **1,178,132 live / 1,178,132 slots**, no holes. The producer
was exactly 1,024 documents ahead of the embedder when it died, which is the work queue's
capacity, so those were lost -- 10,500 submitted, 9,476 durable. The codes index was not
written, and the next open rebuilt it and saved it without being asked.

**One hour was sized on a lie.** The run was sized at 15,000 articles from the 500-article
rate of 4.5 docs/s. Because ingest is quadratic, that rate does not survive contact with a
larger store, and the run was at 2.4 docs/s cumulative when it died. Size ingest runs off the
interval rate at the target scale, not a small-corpus extrapolation.
