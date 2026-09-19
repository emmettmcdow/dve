# Hamming scan throughput

Apple M5 (4 performance + 6 efficiency cores, 6 MB L2, 24 GB RAM), zig 0.15.2,
`-Doptimize=ReleaseFast`. 768-bit codes, 96 bytes each, top-100 kept per query, minimum of
20 runs. Harness: `experiments/hamscan`.

`binrecall` measured this incidentally at 27.8 GB/s and that number was worthless: its 37,689
codes are 3.6 MB against a 6 MB L2, so it measured cache. The planned codes cache for Simple
Wikipedia is ~3.1 GiB. This sweeps from inside cache to well past it.

## The scan is memory-bound. The popcount is free.

Three inner loops over identical loads and an identical access pattern:

- `u64x12` -- twelve `u64` XOR+popcounts. On AArch64 `@popCount` on a `u64` has to move to a
  vector register, `cnt`, and `addv` back.
- `neon` -- six 16-byte lanes, one `cnt` each, no round trip.
- `stream` -- **the control.** Same loads, no popcount, no per-code reduction. It measures
  what this machine streams at, which is the only way to tell whether the other two are at the
  memory wall or at their own compute ceiling.

Milliseconds per query, 8 threads:

| codes | bytes | `u64x12` | `neon` | `stream` (control) |
|---:|---:|---:|---:|---:|
| 100,000 | 9.2 MB | 0.35 | 0.48 | 0.35 |
| 1,000,000 | 91.6 MB | 3.10 | 2.68 | 3.24 |
| 5,000,000 | 457.8 MB | 17.76 | 18.32 | 17.43 |
| 10,000,000 | 915.5 MB | 35.34 | 35.92 | 35.96 |
| 20,000,000 | 1.8 GB | 73.23 | 72.59 | 72.68 |
| **35,000,000** | **3.1 GB** | **130.03** | **128.93** | **120.01** |

**Removing the popcount entirely changes nothing.** Every size at or above 5M lands within a
few percent of the control, at 26-28 GB/s. The scan is bandwidth-bound and the inner loop is
not worth another minute of anyone's time -- a deliberate contrast with the `storedDot`
finding in `storebench.md`, where the inner loop was the whole problem. Measuring beats
guessing in both directions.

## Threads buy 2x, and four is enough

`neon`, milliseconds per query:

| codes | 1 thread | 4 threads | 8 threads |
|---:|---:|---:|---:|
| 5,000,000 | 33.48 | 18.14 | 18.32 |
| 10,000,000 | 76.07 | 35.40 | 35.92 |
| 20,000,000 | 145.68 | 72.22 | 72.59 |
| 35,000,000 | 269.07 | 123.62 | 128.93 |

One thread reaches ~13 GB/s; four reach ~27 and saturate. The extra six efficiency cores add
nothing, because the memory system is already full. **Thread the scan, but size the pool to
the performance cores, not to `getCpuCount()`.**

### Equal slices are wrong on a heterogeneous CPU

The first version gave each thread an equal contiguous slice, and produced a 4-thread column
*slower* than its 8-thread one and a single-threaded 35M figure six times its own trend. On a
4+6 machine an equal split puts identical work on both kinds of core and the join waits for
whichever slice landed on an efficiency core.

Threads now pull 32,768-code chunks off an atomic counter -- small enough to rebalance, large
enough that the atomic disappears. The anomalies went with it. The real implementation needs
the same thing, so this is a design note rather than a harness detail.

## What it means for search

Scan time at 26 GB/s, plus ~6.4 ms for a K=100 stage-2 read (100 cold random reads at 64 us):

| corpus | codes in RAM | scan @4t | + stage 2 |
|---|---:|---:|---:|
| **Simple Wikipedia, as indexed today (~35M)** | 3.1 GiB | 124 ms | **~130 ms** |
| 20M | 1.8 GiB | 72 ms | 78 ms |
| 10M | 916 MB | 35 ms | 42 ms |
| **filtered to >= 6 words (~6.7M)** | 614 MB | ~24 ms | **~30 ms** |
| filtered to >= 9 words (~3.0M) | 275 MB | ~11 ms | ~17 ms |

**As the corpus is indexed today, a query costs ~130 ms. That is not "near instant."**

The only lever is bytes read, because nothing else is binding:

1. **Fewer vectors.** The short-chunk filter in `src/chunking.md` cuts the corpus 5.5x and
   takes the query to ~30 ms. It was parked as a memory optimisation and a quality
   improvement; it is also, unexpectedly, the latency fix. That is the highest-leverage
   change available and it is not in the storage layer at all.
2. **Shorter codes.** 768 bits gives recall@100 = 0.987 (`binrecall.md`) -- far more headroom
   than K=100 needs. A 384-bit code would halve both the memory and the scan, to 1.6 GiB and
   ~62 ms, for some unmeasured recall cost. `binrecall` can measure that cheaply and it is the
   obvious next question.

Note these compose: filtered to >= 6 words *and* 384-bit codes is ~307 MB and ~12 ms.

## What this does not measure

- **mmap.** The array here is heap-allocated and fully resident. A codes file mapped from
  disk behaves the same once warm but has to fault in first, and can be evicted under
  pressure. That is a separate question and it matters for a 3.1 GiB array on a 24 GB machine
  running a ~1 GB embedding model.
- **Concurrent queries.** One query at a time, with all threads on it. Two users at once on a
  saturated memory system will not go twice as fast.
- **Other hardware.** 26 GB/s is this machine. The shape of the conclusion travels; the
  number does not.
