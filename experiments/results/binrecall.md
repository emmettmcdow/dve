# Binary recall: 1-bit codes over real mpnet embeddings

Apple M5, zig 0.15.2, `-Doptimize=ReleaseFast`. Corpus is
`wikitest/wikidata/wikitest-db/mpnet_embedding.db` -- 37,689 real mpnet vectors, 768-dim f32,
from 300 Simple Wikipedia articles. Harness: `experiments/binrecall`.

**The question.** Search is planned in two stages: scan 1-bit codes in RAM for K candidates,
then read those K full vectors from disk and rank them by exact cosine. Stage 2 can only
reorder what stage 1 hands it, so anything stage 1 misses is lost. Recall@K is the share of
the exact cosine top-10 that lands inside the Hamming top-K. It is the go/no-go for the design.

## The variants do not matter. Raw sign wins.

500 queries, full 37,689-vector corpus:

| variant | @10 | @50 | **@100** | @200 | @500 | @1000 | dead bits | balance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **sign** | 0.579 | 0.801 | **0.832** | 0.849 | 0.870 | 0.901 | 6 | 0.143 |
| centered | 0.582 | 0.800 | 0.834 | 0.851 | 0.870 | 0.902 | 0 | 0.032 |
| simhash | 0.552 | 0.793 | 0.829 | 0.848 | 0.870 | 0.901 | 0 | 0.142 |
| centered_simhash | 0.554 | 0.795 | 0.830 | 0.848 | 0.870 | 0.902 | 0 | 0.031 |

All four agree to within 0.005 at every K >= 50. **Take `sign`:** no learned parameters, no
768x768 matvec at ingest, no corpus mean to persist and re-fit as documents arrive.

This was not the expected result. The prior was that coordinate sign would be degenerate on
transformer embeddings, which are known to occupy a narrow cone -- a coordinate whose sign is
the same for most of the corpus spends a bit on nothing. mpnet turns out not to have that
problem at the coordinate level: **only 6 of 768 bits are more than 95% constant**, and
centering, which drives that to 0 and improves bit balance 4.5x (0.143 -> 0.032), buys no
recall at all. Balanced bits are not the same thing as informative bits, and this is the
measurement that says so.

## Recall degrades with corpus size, slowly

Same corpus subsampled, 300 queries each:

| corpus | @10 | @100 | @500 | @1000 | @2000 | K=100 as % of corpus |
|---:|---:|---:|---:|---:|---:|---:|
| 2,500 | 0.602 | 0.892 | 1.000 | 1.000 | 1.000 | 4.0% |
| 10,000 | 0.623 | 0.887 | 0.961 | 1.000 | 1.000 | 1.0% |
| 20,000 | 0.596 | 0.855 | 0.905 | 0.957 | 1.000 | 0.5% |
| 37,689 | 0.586 | 0.842 | 0.879 | 0.909 | 0.963 | 0.27% |

Recall@100 falls 0.892 -> 0.842 while the corpus grows 15x and K shrinks from 4% of it to
0.27%. That is roughly **0.042 per decade of corpus growth**, and it is a far gentler slope
than "vectors crowd into the same Hamming ball" would suggest.

The @500/@1000/@2000 columns degrade faster, but they are pinned at 1.000 for the small
corpora -- K is a large fraction of those -- so their slope is a ceiling artifact. @100 is the
column to read.

**Extrapolating this to the ~35M vectors of full Simple Wikipedia gives recall@100 ~= 0.72,
and that number should not be trusted.** It projects three decades from 1.2 decades of
measurement, which is the exact error corrected three times already in `vec_storage2.md`.
Take it as "probably degrades, probably not catastrophically" and nothing more. Settling it
needs a bigger corpus, which means a wikitest ingest run.

## What recall@100 = 0.83 actually costs

Reranking fixes the *order*, so the user sees the true top-10 minus whatever stage 1 dropped.
At 0.83 that is 8-9 of the right 10 results, correctly ranked, with 1-2 replaced by near-misses.

Buying the rest is expensive, because K is paid for in cold random reads at ~64 us each:

| K | recall @37,689 | stage-2 disk cost |
|---:|---:|---:|
| 100 | 0.832 | 6.4 ms |
| 500 | 0.870 | 32 ms |
| 1000 | 0.901 | 64 ms |
| 2000 | 0.957 | 128 ms |

Going from K=100 to K=1000 buys 7 points of recall for 10x the disk latency. **K=100 is the
right operating point**, and the lever for better recall is a wider code, not a bigger K.

## The scan number, with a caveat that matters

The Hamming scan of 37,689 codes took **0.13 ms**, which is 3.6 MB at ~27.8 GB/s.

**That is a cache measurement, not a memory one.** 3.6 MB fits in L2/SLC on this machine, so
it says nothing about a 3.4 GB array at full corpus scale, where the scan is DRAM-bound.
Read as an upper bound it projects to **>= 122 ms single-threaded at 35M vectors**, which is
not "near instant" and makes threading the scan mandatory rather than an optimisation. The
scan is an independent reduction per slice of the array, so it parallelizes trivially; eight
P-cores should put it in the 15-25 ms range.

Measuring this properly needs a synthetic array well past cache size. That is the next
experiment, and it is independent of everything above.

## What this does not measure

- **Real queries.** Queries here are corpus vectors, the standard ANN-benchmark practice, but
  a real query is a short question whose embedding may sit further from every document vector
  than documents sit from each other. Recall could be worse. `binrecall` has no text mode yet.
- **Scale.** 37,689 vectors against a ~35M target. See the extrapolation warning above.
- **Wider codes.** 2-bit at 192 B/vector would be 6.7 GB at full scale and presumably recovers
  much of the loss. Unmeasured, and the obvious next knob if 1-bit proves too lossy.
