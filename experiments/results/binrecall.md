# Binary recall: 1-bit codes over real mpnet embeddings

Apple M5, zig 0.15.2, `-Doptimize=ReleaseFast`. Corpus is
`wikitest/wikidata/wikitest-db/mpnet_embedding.db` -- 37,689 real mpnet vectors, 768-dim f32,
from 300 Simple Wikipedia articles. Harness: `experiments/binrecall`.

**The question.** Search is planned in two stages: scan 1-bit codes in RAM for K candidates,
then read those K full vectors from disk and rank them by exact cosine. Stage 2 can only
reorder what stage 1 hands it, so anything stage 1 misses is lost. **K is the candidate count,
which is also the number of random disk reads, which is the cost.** Recall@K is the share of
the exact cosine top-10 that survives into those K candidates.

## Read the duplicate-free rows

The corpus contains a great deal of repeated boilerplate -- "Related pages", "References",
"Other websites" -- because `\n` is a sentence delimiter and the median chunk is two words
(see `src/chunking.md`). A typical query has **494 vectors at cosine > 0.99** to it, and that
count barely moves between 0.85 and 0.99: they are the same text, many times over.

**Those queries measure nothing.** When 494 vectors are identical, "the exact cosine top-10"
is an arbitrary 10 of 494 tied candidates, and the Hamming top-K is a different arbitrary
selection from the same tie. Overlap is then `K/494` by chance rather than by quality --
recall 0.2 at K=100 for a code that found every one of them at distance 0. 118 of 300 queries
are like this, and averaging them in drags the headline down by 15 points for a reason that
has nothing to do with the code.

So the harness scores them apart. The `no-dup` rows below are the honest measurement.

## The result: recall@100 is ~0.99, and flat in corpus size

`sign` codes, 300 queries per size:

| corpus | @10 | @25 | @50 | **@100** | @200 | @500 | dup queries |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,500 | 0.708 | 0.920 | 0.969 | **0.990** | 0.995 | 1.000 | 85/300 |
| 10,000 | 0.704 | 0.916 | 0.976 | **0.992** | 0.997 | 0.999 | 92/300 |
| 20,000 | 0.714 | 0.926 | 0.967 | **0.990** | 0.996 | 0.998 | 119/300 |
| 37,689 | 0.693 | 0.907 | 0.957 | **0.987** | 0.995 | 0.998 | 118/300 |

**Recall@100 does not move across a 15x corpus growth.** An earlier version of this file
reported 0.892 -> 0.842 over the same range and read a degradation slope into it. That slope
was the duplicate artifact growing -- more corpus, more chance a query has a copy -- not the
code getting worse. There is no measurable degradation here to extrapolate.

That is a stronger claim than the previous one but it is still a claim about 37,689 vectors
against a ~35M target, and flat over 1.2 decades is not proof of flat over four. It is,
however, no longer a result that *needs* extrapolating to look acceptable.

## The variants do not matter. Raw sign wins.

Full corpus, 300 queries, duplicate-free:

| variant | @25 | @50 | @100 | @200 | dead bits | balance |
|---|---:|---:|---:|---:|---:|---:|
| **sign** | 0.907 | 0.957 | **0.987** | 0.995 | 6 | 0.143 |
| centered | 0.900 | 0.956 | 0.985 | 0.995 | 0 | 0.032 |
| simhash | 0.875 | 0.952 | 0.983 | 0.992 | 0 | 0.142 |
| centered_simhash | 0.882 | 0.953 | 0.983 | 0.995 | 0 | 0.031 |

All four agree to within 0.004 at K >= 100. **Take `sign`:** no learned parameters, no 768x768
matvec at ingest, no corpus mean to persist and re-fit as documents arrive.

This was not expected. Transformer embeddings occupy a narrow cone, so coordinate sign should
have been degenerate. mpnet does not have that problem at the coordinate level: only **6 of
768 bits** are more than 95% constant, and centering -- which drives that to 0 and improves
bit balance 4.5x -- buys no recall at all. Balanced bits are not informative bits.

## A distance threshold cannot replace top-K

`search` today is threshold-shaped: everything above `THRESHOLD` (0.36), then the best N of
those. The natural two-stage version is a loose Hamming cutoff feeding a tight cosine one.
Measured, it does not work:

| hamming <= | share of real matches kept | candidates read | disk @64us |
|---:|---:|---:|---:|
| 250 | 0.323 | 2,107 | 135 ms |
| 275 | 0.579 | 2,876 | 184 ms |
| 300 | 0.947 | 4,418 | 283 ms |
| 325 | 1.000 | 8,895 | 569 ms |

There is no cutoff that is both cheap and complete. But the deeper reason is not that Hamming
is a noisy estimate of angle -- it is that **the cosine floor itself is not selective**:

| cosine > | matches per query | share of corpus |
|---:|---:|---:|
| 0.36 (`THRESHOLD`) | 3,706 | 9.83% |
| 0.46 (`STRICT_THRESHOLD`) | 2,601 | 6.90% |
| 0.65 | 984 | 2.61% |
| 0.85 | 500 | 1.33% |

A tenth of the corpus clears the production threshold. No code can turn a filter that admits
10% of everything into a small candidate list, because the filter is not the thing narrowing
it. Some of that is the duplicate boilerplate, but subtracting the ~494 duplicates still
leaves 8.5%.

**So the candidate list has to be bounded by count, not by quality, and top-K is forced.**
K=100 at 6.4 ms of disk reads, for ~0.99 recall.

### This is also a bug in `search` today

`vstore.search` adds *every* above-threshold vector to a priority queue and then pops
`buf.len` of them. At this corpus that is 3,706 heap inserts per query. At 35M vectors it
would be ~3.4M entries of 40 bytes -- **~136 MB allocated per query** -- to return 10 results.
The threshold was doing the work of a bound and it cannot. A bounded top-K heap fixes it and
is strictly less code.

## The scan number, with a caveat that matters

The Hamming scan of 37,689 codes took **0.13 ms**: 3.6 MB at ~27.8 GB/s.

**That is a cache measurement, not a memory one.** 3.6 MB fits in L2/SLC here, so it says
nothing about a 3.4 GB array at full corpus scale, where the scan is DRAM-bound. Read as an
upper bound it projects to **>= 122 ms single-threaded at 35M vectors**, which is not "near
instant" and makes threading the scan mandatory rather than an optimisation. The scan is an
independent reduction per slice, so it parallelizes trivially; eight P-cores should land in
the 15-25 ms range. Measuring it properly needs a synthetic array well past cache size, and
that is the next experiment.

## Code width: shorter codes plus a bigger K beat a wide code

The scan is memory-bound (`hamscan.md`), so scan time is proportional to code size exactly.
K is paid for separately, in disk reads. That makes the two knobs tradeable, and the trade is
lopsided.

Recall@K on the full corpus, duplicate-free, best variant at each width:

| bits | bytes | RAM @35M | @100 | @200 | @500 | best variant |
|---:|---:|---:|---:|---:|---:|---|
| 768 | 96 | 3.36 GB | 0.987 | 0.995 | 0.998 | `sign` |
| 512 | 64 | 2.24 GB | 0.966 | 0.989 | 0.997 | `centered` |
| **384** | **48** | **1.68 GB** | 0.933 | 0.973 | **0.992** | `centered` |
| 256 | 32 | 1.12 GB | 0.901 | 0.945 | 0.973 | `centered` |
| 192 | 24 | 0.84 GB | 0.833 | 0.901 | 0.948 | `centered` |
| 128 | 16 | 0.56 GB | 0.737 | 0.814 | 0.897 | `centered_simhash` |

Scan time scaled off the measured 124 ms at 96 bytes, plus K x 64 us of stage-2 disk:

| | recall | scan | + disk | total | RAM |
|---|---:|---:|---:|---:|---:|
| 768 bits, K=100 | 0.987 | 124 ms | 6 ms | **130 ms** | 3.36 GB |
| **384 bits, K=500** | **0.992** | 62 ms | 32 ms | **94 ms** | **1.68 GB** |
| 256 bits, K=500 | 0.973 | 41 ms | 32 ms | 73 ms | 1.12 GB |
| 192 bits, K=200 | 0.901 | 31 ms | 13 ms | 44 ms | 0.84 GB |

**768 bits at K=100 is not on the Pareto frontier.** 384 bits at K=500 is better on every
axis at once -- higher recall, 28% less latency, half the memory. The rule is: **shrink the
code and spend the saving on a larger K.** Recall lost to a narrower code is bought back more
cheaply by reading more candidates than by scanning more bytes.

### Which makes raw sign the right code again, for a second reason

Centering wins at every width below 768 -- by 1.4 points at 384 bits, 2 at 256, 4.6 at 128 --
which is the anisotropy argument finally showing up. It only bites once bits are scarce enough
that spending one on a near-constant direction matters.

But it only wins at *small* K. At 384 bits and K=500, the recommended point, `sign` scores
0.992 and `centered` scores 0.991: identical. So the persisted corpus mean -- 3 KB of
parameters that drift as documents arrive and need re-fitting -- buys nothing where it would
actually be used.

**Take the first 384 coordinate signs.** Truncation also matches random projection at this
width (0.919 vs 0.912 at K=100), so there is no matvec at ingest either. Every variant that
costs something has now been measured and none of them earns it.

### The disk model is the weak part of this

`K x 64 us` assumes cold random reads issued one at a time, which is what `full.md` measured
(15,605 IOPS at queue depth 1). Real NVMe does far better with several reads outstanding, and
`experiments/README` deliberately never measured concurrency. If stage 2 parallelizes, K gets
cheaper and the frontier moves *further* toward short codes and large K -- so the
recommendation is robust in direction even though the numbers would move. Worth measuring
before fixing K.

## What this does not measure

- **Real queries.** Queries here are corpus vectors, standard ANN-benchmark practice, but a
  real query is a short question whose embedding may sit further from every document than
  documents sit from each other. Recall could be worse. `binrecall` has no text mode yet.
- **Scale.** 37,689 vectors against ~35M. Flat over the measured range is encouraging, not
  conclusive.
- **Wider codes.** Measured downward rather than upward, above. 2-bit codes were never worth
  trying: 768 one-bit codes already reach 0.998 at K=500, so there is no headroom to buy.
- **Read concurrency in stage 2**, which is what the disk half of the frontier rests on.
