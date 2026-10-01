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

---

# 2.72M vectors, after the coherence work [2026-10-01]

Apple M5, `-Dllama -Doptimize=ReleaseFast`. Same `--sample 1` ordering as the run above, so
the first 9,476 articles are the *same articles* and the two curves overlay point for point --
the vector count at 2,000 articles is 126,732 in both.

Raw output in `raw/wikitest-llama-fixed-20261001-1201.txt`, per-interval samples in
`raw/wikitest-llama-fixed.csv`.

## Ingest is no longer quadratic

| articles | before s/doc | after s/doc | speedup |
|---:|---:|---:|---:|
| 3,000 | 0.120 | 0.113 | 1.1x |
| 4,000 | 0.225 | 0.106 | 2.1x |
| 5,000 | 0.385 | 0.106 | 3.6x |
| 6,000 | 0.548 | 0.127 | 4.3x |
| 8,000 | 0.633 | 0.112 | 5.6x |
| 9,000 | 0.657 | 0.112 | 5.8x |
| 10,000 | 0.758 | 0.146 | **5.2x** |

The speedup column is not the result. The *shape* is: `before` climbs without bound, `after`
does not move. Carried out to 22,000 articles and a 10.4 GB store:

| articles | store | vec/s | s/doc |
|---:|---:|---:|---:|
| 2,000 | 1.0 GB | 1,430 | 0.091 |
| 6,000 | 2.9 GB | 1,124 | 0.127 |
| 10,000 | 4.8 GB | 955 | 0.146 |
| 14,000 | 6.8 GB | 1,006 | 0.126 |
| 18,000 | 8.7 GB | 998 | 0.122 |
| 22,000 | 10.6 GB | 1,001 | 0.132 |

Flat across an eleven-fold growth in the store, oscillating 0.080-0.158 s/doc with no trend.
Total: **22,000 articles, 2,718,260 vectors, 43m56s**, 10.4 GB, 2.2 GB peak RSS during ingest.

The old run never reached 22,000; it was 67 minutes in at 10,000 articles and still slowing
when a harness bug killed it. This run passed 10,000 at 17 minutes.

## Search at 2.3x the vectors

| | 1.18M (previous) | 2.72M |
|---|---:|---:|
| store | 4.5 GB | 10.4 GB |
| codes | 56.7 MB | 130.8 MB |
| open | 416 ms | 1.63 s |
| **search** | 2.2-6.9 ms | **2.1-4.8 ms** median |
| exhaustive scan | 710 ms | 1.79 s |
| **speedup** | ~250-300x | **~390-1280x** |
| peak RSS | 653 MB | **708 MB** |
| recall@10 | 0.950 | **0.967** |

Search did not get slower. Stage one is a linear scan of 130 MB of codes, which is still
nothing, and stage two reads a fixed `candidates` worth of vectors regardless of corpus size
-- so the only thing that grew is the code scan, and at these sizes it is not the cost. Query
embedding, at 11-15 ms warm, remains several times the search.

Memory is the headline for the original goal: **708 MB resident for 2.72M vectors over a
10.4 GB store**, and 547 MB of that is the llama model, not the index.

## A prediction that did not come true

The previous write-up called `createId`'s whole-manifest rewrite the next bottleneck, and this
run was sized partly to catch it emerging. It did not. The manifest reached 693 KB at 22,000
paths, written and `F_FULLFSYNC`ed once per new document, and the curve above is flat -- so at
this scale it is below the noise, even after the fsync was deliberately made *stronger*. It
is still O(paths) per path and still wrong at 283k, where the manifest would be ~9 MB; it is
just not wrong yet, and the number to beat is now measured rather than guessed.

## Coherence held

Audited straight off the files afterwards:

```
manifest : next_id 22001, 22000 paths
store    : 2718260 slots, 2718260 live, 22000 doc_ids, 0 freed, max doc_id 22000
paths with zero live vectors : 0
live doc_ids with no path    : 0
next_id > max doc_id         : YES
```

22,000 documents through the fast path -- every one of them skipping `vecsForDoc` on the
strength of "no path means no vectors" -- and the two files still agree exactly.

## What this still does not measure

- **Cold.** Out of scope by decision, and the 1.63 s open and 1.79 s exhaustive scans here
  were all warm.
- **Re-embed.** Every document in this run was new, which is the case the shortcut fixes. An
  edit to an existing document still walks the whole store, and at 10.4 GB that is ~1.7 s.
  That is the next piece of work, and it is what a doc_id -> slots index is for.
- **The full corpus.** 22,000 of 283,547 articles. At this rate the remainder is ~9.5 hours of
  embedding, but 36.9M chunks is ~151 GB of store, so disk is now the binding constraint
  rather than time.
