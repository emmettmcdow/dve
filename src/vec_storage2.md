# Vec Storage 2

`src/vstore.zig`. Page-aligned, disk-resident replacement for `vec_storage.zig`.

## Status (2026-09-09)

Store is complete and green: `zig build test-vstore` is 56/56 (44 store + 12 `pfile`),
full `zig build test` is 152/154 with the 2 pre-existing skips. Nothing committed yet;
`vstore.zig`, `pfile.zig`, and this file are untracked, `build.zig` is modified.

**Not yet wired into `vector.zig`** -- both stores coexist until the cutover at the bottom
of this file. Every API `vector.zig` touches exists in `vstore.zig`; the diff is naming and
signatures only.

Design target is **indexing an entire filesystem**: ~20M vectors, ~82 GB on disk at 768xf32.
Optimise for `put` and `search`; `get`/`rm` are barely used (`vector.zig` uses a vector id in
exactly two places and discards `put`'s return value).

**The storage layer survived the measurements.** Page-aligned chunks, trailer metadata, CRCs,
and the libc IO path all held; every conclusion that got overturned lived above them. The open
question moved up a layer, to the index -- see "The index layer" below, which is deliberately
undecided.

Two independent tracks, neither blocking the other:

- **Storage:** slot reuse -> cutover to `vector.zig` -> wikitest. Settled work, no open design
  questions, and unaffected by whatever the index turns out to be.
- **Index:** binary-recall experiment -> Hamming scan throughput -> pick an index. Both are
  measurements, neither touches `vstore.zig`.

Priorities within storage were re-derived from `experiments/` after conclusions drawn at
65k-vector scale failed to survive the jump to filesystem scale.

File IO goes through `src/pfile.zig`, a thin shim onto libc: `open`, `close`, `pread`,
`pwrite`, `lseek`, `ftruncate`, `fsync`, `fcntl`. No `std.fs.File` anywhere in the store.
The build target needs `link_libc = true` -- currently only `test-vstore` has it.

## Settled design

- **4096-byte pages.** MacOS uses 16KB memory pages but 4KB disk blocks, and most other
  architectures use 4KB for both, so we align to the lowest common denominator.
- **Header is page 0 and is immutable.** Magic, format version, endianness, and the layout
  constants only -- no counters. Nothing mutable in the header means no window where a crash
  leaves it disagreeing with the pages behind it, and no fsync ordering to get right.
  `vec_n` and the slot high-water mark are derived by scanning metadata trailers at open.
- **Chunk = vectors from the front, 32-byte metadata entries in a trailer at the back.**
  Keeps vector 0 at page offset 0, keeps the vector region contiguous for scanning, and
  makes a future 1-bit-code build a straight stride.
- **Packing solves for vectors and metadata together.** `k = PAGE / (stride + 32)`. Picking
  a vector count first and hoping the remainder fits the metadata fails badly for small
  vectors: 4096/12 = 341 vectors leaves 4 bytes for 341 entries. When `k` comes out 0 the
  vector spans `P = ceil((stride + 32) / PAGE)` pages instead. Exactly one of `k` and `P` is
  ever greater than 1.
- **Stride is rounded up to 16** so every vector in a chunk is SIMD-aligned, not just the
  first. Free at 768 dimensions, where 3072 is already a multiple of 16.
- **Checksums:** crc32 per vector covering the vector bytes plus the first 24 bytes of its
  metadata, deliberately excluding `flags`. `rm` can then flip one byte without re-reading
  and re-hashing the vector, while a torn write that leaves `occupied` set over half-stale
  data is still caught. A one-byte flag update cannot itself tear.
- **ids are stored, not inferred.** `vec_id` lives in every metadata entry. v1's allocator
  happens to hand out `id == slot`, but that is an allocator detail, so slots can later be
  reused or relocated with no format change. Reuse turns out not to need an index (priority 1),
  but keeping the id on disk is what makes a future relocating layout possible.
- **Append-only.** `rm` tombstones; slots are never reused. See the caveat below.
- **IO is libc, not `std.fs.File`.** POSIX positional IO is a stable interface; the standard
  library's buffered Reader/Writer is not, and rewriting it is what broke `vec_storage.zig`
  at 0.15. `pfile.zig` owns the three things the standard library had been hiding: partial
  transfers (every call loops), EINTR (every call retries), and Darwin's `fsync`, which
  returns before the drive flushes its own write cache -- `F_FULLFSYNC` is the one that
  actually commits, and it is what `flush()` now issues. `open` and `fcntl` are declared
  variadic because on AArch64 macOS variadic arguments go on the stack while fixed ones go in
  registers, so a fixed-parameter declaration would pass the mode in the wrong place.
- **No caching, one mutex.** Every operation goes to disk; the mutex covers the whole public
  surface. The read/write split is already visible in the API, so an RwLock is a drop-in.
- **`save`/`load` are gone.** There is no in-memory copy to write back, so they were never
  really save and load -- `flush()` is a barrier (fsync) and the filename moved into `Opts`.
  Real transactions are a later want.

## Layout, measured

| config | vec bytes | stride | k | P | slack | waste |
|---|---|---|---|---|---|---|
| `N=3, f32` (tests) | 12 | 16 | 85 | 1 | 16 | ~0% |
| `768 x i8` | 768 | 768 | 5 | 1 | 96 | 2.3% |
| `768 x f16` | 1536 | 1536 | 2 | 1 | 960 | 23.4% |
| **`768 x f32` (prod)** | 3072 | 3072 | 1 | 1 | 992 | **24.2%** |
| `1016 x f32` | 4064 | 4064 | 1 | 1 | 0 | 0% |
| `1536 x f32` | 6144 | 6144 | 1 | 2 | 2016 | 24.6% |

The ~24% is the price of page-aligning every vector when the size does not divide the page.
Accepted: storage is the cheapest resource, quantization shrinks it, and the interface hides
the layout well enough to change it later. The slack is where metadata and checksums live,
with ~880 bytes still free at 768xf32 -- enough for a 96-byte 1-bit code.

## Known v1 costs

- **Unbounded growth on re-embed.** `vector.zig` `replaceVectors` puts the new vectors then
  removes the old on every re-embed, so an actively edited document accumulates tombstones
  forever. The old implementation reused holes and did not. Not a correctness problem, but it
  is the only real defect -- see priority 1 below.
- **Open scans every trailer.** One 32-byte `pread` per chunk at startup, O(n) per open, and
  measured at **30.5 us/chunk cold** -- ~10 minutes at 20M vectors. This is the single worst
  scaling problem in the design. See priority 2.
- **`put` costs two pwrites**, `get` one pread, `rm` one pread plus a one-byte pwrite.

## Priority 1: slot reuse via an in-memory free list

`rm` tombstones and slots are never reused, so `replaceVectors` grows the file forever on an
actively edited corpus. This is the only real v1 defect.

**It needs no index and no compaction pass.** `vec_id`s are ephemeral: `SearchResult` carries
path/start_i/end_i/similarity and no id, `note_id_map` has no `VectorID` at all, and inside
`vector.zig` ids appear only as `rm(old_v.id)` within `replaceVectors` plus one test helper --
`put`'s return is discarded. No id is persisted, exposed to consumers, or held across a process
boundary.

So: **build a free list during the open scan.** The scan already reads every metadata entry, so
the list is free -- no extra IO, no new file, no format change, no generation stamping, no
stale-index recovery path. `put` pops a free slot; `get(id)` stays pure arithmetic against
`id == slot` with zero resident memory. A bitmap costs 8 KB at 65k slots, 128 KB at 1M.

Note the inversion: the open scan is what makes the sidecar index unnecessary. Keeping the scan
is what lets us delete the index, not the other way round.

Two things this requires:

- **Document the id contract:** an id is valid until the vector is removed, and a removed id may
  be reissued to a later `put`. That is a real weakening of today's never-resurrected guarantee.
- **`replaceVectors` ordering becomes load-bearing.** It puts the new vectors *then* removes the
  old, so old ids are still live while the puts run and no `put` can hand back a slot the caller
  is still holding. Safe today, but incidental -- it needs a comment at both ends saying so, or a
  reordering breaks reuse in a way no current test would catch.

Reuse stops the file *growing*; it does not *shrink* one already bloated. `compact()` is the
shrink tool and stays deferred, probably forever -- it is only worth it if a corpus churns down
hard and the slack is worth reclaiming.

## Priority 2: batching the open scan  [REVISED -- was "not doing"]

**This section previously said "investigated and dropped." That was wrong, and the reason it
was wrong is instructive.** The 2x figure it cited was measured entirely in RAM, on a 256 MiB
file. At filesystem scale the file is several times larger than RAM, nothing caches, and the
same comparison is worth 40x+. See `experiments/results/full.md` -- 32 GiB, F_NOCACHE, on a
file deliberately larger than RAM so caching cannot interfere.

Measured, cold (Apple M5, 24 GB RAM):

| pattern | rate | per unit |
|---|---|---|
| sequential, 1 MiB blocks | 5,699 MiB/s | -- |
| sequential, 128 KiB blocks | 2,474 MiB/s | -- |
| sequential, 4 KiB blocks | **135 MiB/s** | 29 us/block |
| 32 B trailer per 4 KiB chunk (today's scan) | -- | **30.5 us/chunk** |
| random 4 KiB read | 15,605 IOPS | **64 us** |

Warm, for contrast: 10,978 MiB/s sequential, 1.6 us per random read. **The warm and cold
regimes differ by 40x on the pattern we care about**, which is exactly why the earlier
extrapolation from a warm 256 MiB measurement was worthless.

Note the trailer scan costs the same as reading every byte in the file (1.06x of a full 4 KiB
scan). At 768xf32 each 32-byte trailer sits in its own 4 KiB block, so "read only the metadata"
touches every block anyway. Reading 0.8% of the bytes buys nothing.

Projected to the 20M-vector target (81.9 GB on disk at 768xf32):

- **today's scan:** 20M x 30.5 us = **~10 minutes** at every cold open.
- **batched at 1 MiB:** 78,125 MiB / 5,699 MiB/s = **~14 seconds**.
- **45x**, not 2x.

Fourteen seconds is still far too slow for app launch, so batching alone does not save the
scan -- the sidecar index does, and batching is what makes the unavoidable cold rebuild
tolerable. Both are needed at this scale. Use 1 MiB blocks, not the 128 KiB the warm curve
suggested: warm was cache-bound above 1 MiB, cold is still climbing at 1 MiB.

## The disk-read budget, and what it implies

The crossover the experiment was built to find:

> **An index must probe fewer than 1.07% of the corpus to beat a full sequential scan.**

That sounds permissive until it is turned into a latency budget. At 64 us per cold random read,
an interactive query can afford **~150 reads per 10 ms**. So:

- Probing 1% of 20M vectors is 200,000 random reads = **12.8 seconds**. Any index whose probe
  count scales with corpus size loses outright, however good its recall.
- The disk read count per query must be a small constant -- order 100 candidates -- not a
  fraction of the corpus.

This is the measured argument for the in-memory quantized cache, and it is stronger than the
intuition it confirms: **stage one cannot touch the disk at all.** The two-stage design is
"scan codes in RAM, then verify ~100 candidates on disk" = codes-scan + ~6 ms, and there is no
disk-resident alternative that competes. The codes scan itself is still unmeasured; that is the
next experiment.

## The index layer -- open, nothing committed

The measurements settled the *storage* question and moved the open one up a layer. Nothing
below is decided; this records the discussion so it does not have to be re-derived.

### What the disk experiment did and did not prove

It proved that **disk-resident random access loses**, at 64 us per cold read. It says nothing
about an in-memory index: HNSW over 20M nodes does ~200-500 random *RAM* accesses at ~100 ns,
around 50 us, which would beat a ~20 ms linear scan by orders of magnitude. The linear-scan
hypothesis is still open, and its real opponent is not "k-means on disk" -- that is dead -- but
**IVF over the same 1-bit codes, in RAM**. Linear scan is bandwidth-bound, so probing 1% of
clusters scans 19 MB instead of 1.9 GB. That is an add-on to the same codes, not a rewrite.

### Candidates

| | query | insert | memory @20M | build |
|---|---|---|---|---|
| linear scan over codes | ~20-60 ms (est, unmeasured) | append 96 B, O(1) | 1.9 GB | none |
| IVF over the same codes | ~100x less bandwidth at nprobe=1% | assign + periodic re-fit | 1.9 GB + centroids | k-means |
| HNSW | ~50 us (est) | O(log n), graph mutation + locking | 1.9 GB + ~2.5 GB edges | incremental |

Linear scan wins **insert** outright, which matters more here than it looks: indexing a
filesystem means 20M initial inserts and then continuous churn as files change. It also has no
build step, no recall cliff, and no tuning. It loses on query latency if the corpus is large
enough, and where that line sits is unmeasured.

### What would disprove the linear-scan hypothesis

Ranked by likelihood x damage:

1. **Binary recall.** The two-stage design works only if the true top-10 by cosine sit inside
   the Hamming top-K. Reranking reorders candidates; it cannot recover one stage 1 never
   surfaced. Unmeasured, and a go/no-go gate for the whole architecture. **Must be tested at
   two corpus sizes** -- binary recall degrades as vectors crowd into the same Hamming ball, so
   a good result at 10k documents says little about 20M. That is the same measure-one-scale-
   reason-about-another error that has already been corrected three times in this file.
2. **Single-core scan throughput.** The ~20 ms estimate is 1.9 GB over an assumed ~100 GB/s.
   One P-core typically reaches ~30-50 GB/s of package bandwidth, and XOR+popcount at ~3 NEON
   ops per 16 bytes may be compute-bound rather than bandwidth-bound. Could be 40-60 ms.
3. **Residency.** 1.9 GB must stay resident alongside a ~1 GB embedding model on a 24 GB
   machine. macOS compresses under pressure, and anything that reaches swap turns a memory scan
   into 64 us disk reads -- a cliff, not a slope. At 200M vectors (19 GB) it cannot fit at all.
4. **The latency target**, which is a product decision, not a measurement: under 10 ms and
   linear scan loses on its own numbers; at 100 ms it wins comfortably.

### Keeping the index separate from storage

The goal is swappable indices without premature abstraction. The resolution:

**Separation comes from dependency direction, not from an interface type.** No index vtable, no
`Index` trait, nothing designed from imagination before a second implementation exists. Instead
one rule: index code lives outside `vstore.zig` and depends on it; `vstore.zig` never learns
what a candidate or a code is. That alone makes indices swappable -- a second index is a second
module, and choosing between them is one line in `vector.zig`, later a comptime parameter with
no runtime cost.

This follows the write-long-then-compress approach deliberately. There are currently **zero**
index implementations; inventing the interface now would be designing against imagination.
Build linear-scan concretely and verbosely, and when IVF arrives the shared shape will be
visible and can be compressed out. What is needed to *compare* implementations is not a shared
interface but a benchmark harness, and `experiments/` is already that.

**One option worth not deciding yet:** the ~880 bytes of slack per chunk at 768xf32 could hold
a 96-byte code. Storing codes there makes them durable and free (the slack is wasted anyway),
but couples the storage format to one index's needs. The middle path is for `vstore` to reserve
the slack as an opaque per-slot scratch region it persists and never interprets -- storage
provides durable bytes, the index decides their meaning. Nothing is lost by deferring this: the
slack already exists and is already unused.

## Deferred, roughly in order

1. **Slot reuse** (priority 1). Fixes unbounded growth, and keeps the in-memory codes array
   dense -- a dead slot costs query bandwidth on every search, forever.
2. **Batched open scan** (priority 2), 1 MiB reads. 45x measured, cold.
3. **Sidecar index**, back on the list: even batched, a cold open at 20M vectors is ~14 seconds,
   so the scan cannot be the startup path. Must also persist the quantized codes -- recomputing
   them means re-reading 82 GB. Still never inside the data file, still never fatal to lose.
4. **1-bit quantized cache** of the whole DB in memory: scan the codes, then confirm ~100
   candidates on disk. The ~880 bytes of slack per chunk at 768xf32 has room for a 96-byte code,
   so this needs no format change. The measurements make this mandatory, not optional.
5. **RwLock**, once contention is real.
6. **Transactions.**
7. **`compact()`**, only if reclaiming space from a churned-down corpus turns out to matter.

**On the sidecar index, twice reversed.** It was first justified by two grounds, then dropped
when both dissolved (reuse needs only an in-memory free list; the scan looked cheap). The
measurements put it back: the scan is cheap only warm, and at filesystem scale a cold open is
~10 minutes. It returns as a cache of *codes plus structure*, still never in the data file, and
still never fatal to lose.

The lesson worth keeping: all three reversals came from measuring one scale and reasoning about
another. Numbers in this file now cite `experiments/` or say they are estimates.

## Open questions

- **No benchmark on a real corpus yet.** The wikitest run is the check that matters and has
  not happened, because it needs the cutover.
- **The in-memory Hamming scan is unmeasured.** ~20 ms/query at 20M vectors is 1.9 GB divided
  by an assumed ~100 GB/s of memory bandwidth. It is the load-bearing number for the whole
  search design and deserves the same treatment the disk numbers just got -- next experiment.
- **How many sentences a real filesystem produces**, which sets whether 20M is the right target
  at all. Everything above is sized off an estimate of ~100-500 sentences per document.

## Cutover checklist for `vector.zig`

- `note_id` -> `doc_id`; `vecsForNote` -> `vecsForDoc`; `VecForNoteEntry` is gone (`Row`).
- `init(alloc, dir, .{})` + `load(path)` -> `init(alloc, dir, .{ .path = path })`.
- `save(path)` -> `flush()`.
- `get` returns `!?Row`; `getVec(id, out: *Array)` writes into the caller's buffer.
- `put(.{ .doc_id, .start_i, .end_i }, vec: *const Array)` -- the array form, not `@Vector`,
  since `@Vector(768, f32)` is padded to 4096 bytes and aligned to 4096.
- `get`/`getVec`/`search` now take `*Self`.
- `rm` returns `!void` (was `Error!void`); `rmByDocId` can now fail.
