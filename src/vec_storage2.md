# Vec Storage 2

`src/vstore.zig`. Page-aligned, disk-resident replacement for `vec_storage.zig`.

## Status (2026-09-10)

Store is complete and green: `zig build test-vstore` is 63/63 (51 store + 12 `pfile`),
full `zig build test` is 226/228 with the 2 pre-existing skips. Committed, including
slot reuse. `zig build test` did not run the store's tests until 43ce0e3 wired
`test-vstore` into the aggregate, which is why the total jumped from 154 to 228.

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

- **Storage:** slot reuse (**done**, af581da) -> cutover to `vector.zig` -> wikitest. Settled
  work, no open design questions, and unaffected by whatever the index turns out to be.
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
- **ids are stored, not inferred.** `vec_id` lives in every metadata entry, which is what
  paid for slot reuse: the generation is already on disk in bytes every read path fetches.
  Keeping the id on disk is also what makes a future relocating layout possible.
- **Slots are reused, and ids carry a generation.** `rm` tombstones on disk and pushes the
  slot onto an in-memory free list that the open scan rebuilds. An id is
  `(generation << 32) | slot`, so refilling a slot issues a new id and a stale one still
  resolves to nothing. See "Slot reuse" below.
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

- **Open scans every trailer.** One 32-byte `pread` per chunk at startup, O(n) per open, and
  measured at **30.5 us/chunk cold** -- ~10 minutes at 20M vectors. This is the single worst
  scaling problem in the design, and now the top priority. See below.
- **`put` costs two pwrites**, `get` one pread, `rm` one pread plus a one-byte pwrite. `put`
  also called `lseek` on every insert to check whether the file needed extending; the store
  now tracks the chunk count itself (47cf2f7).
- **`put` still grows the file one 4 KB chunk at a time.** A real cost at 20M inserts, but it
  belongs with the batched-scan work, not with reuse.
- **The metadata write is not atomic.** Contained by the crc rather than prevented. See below.

## Slot reuse  [DONE -- af581da, was "priority 1"]

`rm` tombstoned a slot and nothing ever claimed it again, so `replaceVectors` -- which replaces
a document's vectors on every re-embed -- leaked a slot per sentence, forever. That was the only
real v1 defect.

**It needed no index and no compaction pass.** `vec_id`s are ephemeral: `SearchResult` carries
path/start_i/end_i/similarity and no id, `note_id_map` has no `VectorID` at all, and inside
`vector.zig` ids appear only as `rm(old_v.id)` within `replaceVectors` plus one test helper --
`put`'s return is discarded. No id is persisted, exposed to consumers, or held across a process
boundary.

**The free list is built during the open scan.** The scan already reads every metadata entry, so
a slot that is used but not occupied is exactly a hole and the list costs no extra IO, no new
file, and no format change. `put` pops before it extends; `get(id)` stays pure arithmetic. It is
a LIFO stack of ids, 8 bytes per *hole* rather than per slot.

Note the inversion: the open scan is what makes the sidecar index unnecessary for reuse. Keeping
the scan is what lets us delete the index, not the other way round.

### Generations, and why they were free

Reuse without them is silently wrong: after slot 7 is freed and refilled, a caller holding the
old id 7 would `get` someone else's row and `rm` someone else's vector, and `MultipleRemove`
would stop being detectable. So an id is now `(generation << 32) | slot`.

The alternative on the table was a separate id-to-slot map with monotonically increasing ids.
Generations won on every axis:

| | generation-tagged ids | id-to-slot map |
|---|---|---|
| memory @20M | 8 B x *holes* | 8-16 B x *every slot*, ~320 MB, resident forever |
| `get(id)` | arithmetic, as today | hash lookup |
| disk format | unchanged -- `vec_id` was already `u64` | unchanged |
| extra IO | none; `scan` reads the trailer anyway | none; same scan rebuilds it |
| stale-id check | one comparison on metadata already in hand | map miss |

The decisive evidence was in the tests. Three existing cases assert things that were true only
because the store was append-only -- `"ids stay distinct across a delete"`, `"reopen: tombstones
survive..."`, and churn cycle 3. **With generations all three pass unchanged**, because a reused
slot yields a different id. Without them all three become false and the store quietly loses a
guarantee it documents. All 56 tests that existed before reuse still pass untouched.

A slot whose generation is exhausted (2^32 reuses) is retired rather than wrapped, which removes
ABA from consideration entirely.

### Crash consistency is unchanged

`put` still writes the vector before the metadata. A crash between them leaves the reused slot's
old trailer standing -- used, not occupied -- so the next scan puts it straight back on the free
list and the half-written vector bytes are never read. A crash after the metadata write leaves
the slot occupied at the new generation. Both states are correct with no recovery path.

### The id contract

An id is valid until its vector is removed. A removed id is never reissued; the slot behind it
may be, under a new id. `get`/`getVec`/`rm` on a stale id behave exactly as before -- `null`,
`NoSuchVector`, `MultipleRemove` -- whether or not the slot has since been refilled. That is a
real weakening of the old never-resurrected guarantee, and it is the reason generations exist.

**There is no ordering contract between `put` and `rm`.** An earlier draft of this section said
`replaceVectors`'s put-before-remove ordering was load-bearing. It is not, and the claim was a
leftover from the pre-generation design that should have been re-derived once generations
existed. The invariant that makes ordering irrelevant:

> A `put` can only claim a slot that is already on the free list, and a slot only reaches the
> free list by being removed. So no interleaving of `put` and `rm` can make a `put` land on a
> live vector, and a stale id resolves to nothing whenever it went stale.

This matters beyond tidiness: `put` and `rm` are both public, so a hidden sequencing requirement
between them would have been an API defect, not a caller's problem. There is a test --
`"reuse: put and rm can be interleaved in any order"` -- that runs the same replace-a-document
workload three ways (put-then-rm, rm-then-put, interleaved) and requires identical results.

**But the ordering does matter for density, in the opposite direction to what that comment
claimed.** Putting the new rows *before* removing the old leaves the old slots live while the
puts run, so the new rows cannot reuse them: steady state is two generations of every document
rather than one. The vstore churn test shows `slot_n` settling at 16 for 8 live vectors.

`replaceVectors` has been reordered to remove first (93dd20e). This pays off immediately rather
than at cutover -- `vec_storage.zig` also reuses slots (first-fit over unoccupied ones) and
grows off the live count, so putting first made `vec_n` peak at 2x during every re-embed and
forced a doubling. `"embedText re-embedding a document reuses its slots"` re-embeds a
20-sentence document six times against the default capacity of 32 and requires no growth;
with the old ordering it reports capacity 64.

The reorder is also the better failure mode. If a `put` fails partway, the document is left
under-indexed and a re-embed fixes it. Putting first and failing partway leaves the old rows
*and* some new ones covering the same offsets -- a state `validate` rejects as overlapping.

Reuse stops the file *growing*; it does not *shrink* one already bloated. `compact()` is the
shrink tool and stays deferred, probably forever.

The payoff is bigger than disk. Every whole-store read path -- `search`, `vecsForDoc`,
`validate` -- walks chunks up to the high-water mark regardless of how many slots are live, and
the planned in-memory codes array is scanned linearly on every query. A dead slot costs query
bandwidth on every search, forever, so keeping the slot space dense is a search optimisation
that happens to also stop the file growing.

## Priority 1: batching the open scan  [REVISED twice -- was "priority 2", before that "not doing"]

**This section previously said "investigated and dropped." That was wrong, and the reason it
was wrong is instructive.** The 2x figure it cited was measured entirely in RAM, on a 256 MiB
file. At filesystem scale the file is several times larger than RAM, nothing caches, and the
same comparison is worth 40x+. See `experiments/results/full.md` -- 32 GiB, F_NOCACHE, on a
file deliberately larger than RAM so caching cannot interfere.

Measured, cold (Apple M5, 24 GB RAM):

| pattern | rate | per unit |
|---|---|---|
| sequential, 4 MiB blocks | **6,636 MiB/s** | -- |
| sequential, 1 MiB blocks | 5,699 MiB/s | -- |
| sequential, 128 KiB blocks | 2,474 MiB/s | -- |
| sequential, 4 KiB blocks | **135 MiB/s** | 29 us/block |
| 32 B trailer per 4 KiB chunk (today's scan) | -- | **30.5 us/chunk** |
| random 4 KiB read | 15,605 IOPS | **64 us** |

The 4 MiB row is from the 2026-09-10 re-run, which also re-measured every other row: all four
cold numbers reproduced within 3.3% two days later, which is this machine's run-to-run spread
and is why the differences under 5% in the block sweep below are not read as signal. (The warm
numbers move more, up to 12%, as cache-resident measurements do.) See "Block size" below for
why the sweep stopped at 4 MiB.

Warm, for contrast: 10,978 MiB/s sequential, 1.6 us per random read. **The warm and cold
regimes differ by 40x on the pattern we care about**, which is exactly why the earlier
extrapolation from a warm 256 MiB measurement was worthless.

Note the trailer scan costs the same as reading every byte in the file (1.06x of a full 4 KiB
scan, 1.07x on the re-run). At 768xf32 each 32-byte trailer sits in its own 4 KiB block, so "read only the metadata"
touches every block anyway. Reading 0.8% of the bytes buys nothing.

Projected to the 20M-vector target (81.9 GB on disk at 768xf32):

- **today's scan:** 20M x 30.5 us = **~10 minutes** at every cold open.
- **batched at 4 MiB:** 78,125 MiB / 6,636 MiB/s = **~12 seconds**.
- **52x**, not 2x.

Twelve seconds is still far too slow for app launch, so batching alone does not save the
scan -- the sidecar index does, and batching is what makes the unavoidable cold rebuild
tolerable. Both are needed at this scale.

### Block size: 4 MiB, and the curve is flat above it  [measured 2026-09-10]

The 1 MiB figure above was the largest block the original sweep tried, so it was a floor
rather than an optimum -- the cold curve was still climbing when the measurements stopped.
Sweeping past it (`./run.sh --only seq --regime cold --blocks 1M,2M,4M,8M,16M,32M,64M`, same
32 GiB F_NOCACHE file):

| block | cold MiB/s | 81.9 GB scan | vs. 1 MiB |
|---|---|---|---|
| 1 MiB | 5,510 | 14.2 s | 1.00x |
| 2 MiB | 6,384 | 12.2 s | 1.16x |
| **4 MiB** | **6,636** | **11.8 s** | **1.20x** |
| 8 MiB | 6,693 | 11.7 s | 1.21x |
| 16 MiB | 6,337 | 12.3 s | 1.15x |
| 32 MiB | 6,634 | 11.8 s | 1.20x |
| 64 MiB | 6,627 | 11.8 s | 1.20x |

**The knee is at 4 MiB and there is nothing above it.** 4 -> 8 MiB buys 0.6%, and the spread
across 8..64 MiB is 5% with no trend -- the same run-to-run variance that puts today's 1 MiB
number 3% below the 5,699 MiB/s measured on 2026-09-08. So the honest reading is that the
device saturates somewhere around 4 MiB at roughly 6.6 GiB/s single-threaded, and bigger
batches are free to want but buy nothing.

Take **4 MiB**: it captures the whole win, and a scan buffer is resident memory competing with
a ~1 GB embedding model on a 24 GB machine, so there is no reason to pay 64 MiB for 0%. The
*warm* curve gives a second reason not to: it peaks at 1-4 MiB (12,291 MiB/s) and falls to
8,418 MiB/s by 16 MiB, so an oversized block is not merely neutral once the file does fit in
cache -- which is the regime a small store, or the tail of a big one, actually runs in.

Two things this does *not* say. **Bigger blocks are not the remaining 2x.** 6.6 GiB/s
single-threaded is likely a queue-depth limit rather than a bandwidth one -- one thread with
one pread outstanding leaves the device idle between syscalls -- so the next real gain is
concurrent reads, which `experiments/` deliberately does not measure yet (README: "concurrency
is a separate question and would confound this one"). And **11.8 s is an IO ceiling, not a
build time.** The initial index build also quantizes every vector it reads; to stay IO-bound
at 4 MiB the 768xf32 -> 96-byte code path has to sustain ~6.6 GiB/s on one core. If it does
not, the batch size stops mattering and the scan becomes compute-bound -- which is the same
question as the unmeasured Hamming scan throughput below, approached from the other side.

## The disk-read budget, and what it implies

The crossover the experiment was built to find:

> **An index must probe fewer than 0.92% of the corpus to beat a full sequential scan.**

That was 1.07% before the block sweep. The crossover is a ratio of a full scan to one
random read, so making the scan 1.2x faster tightened it by the same factor -- the budget
for an index got *smaller* as a result of the batching win, not larger.

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

## Limits worth publishing

**A store holds at most 2^32 = 4,294,967,296 vectors.** An id spends 32 bits on the slot index,
so that is the hard cap and `put` returns `StoreFull` at it. The cap is on slots, so what it
means in bytes depends on the config:

| config | bytes/slot | max file | of which vector payload |
|---|---:|---:|---:|
| **768 x f32 (prod)** | 4096 | **16.0 TiB** (17.6 TB) | **12.0 TiB** (13.2 TB) |
| 768 x f16 | 2048 | 8.0 TiB | 6.0 TiB |
| 768 x i8 | 819.2 | 3.2 TiB | 3.0 TiB |

The gap between file and payload is the ~24% page-alignment slack. Against the ~20M-vector
design target this is ~215x of headroom, so it is a datapoint for users rather than a constraint
on us. Two notes: the cap counts the slot high-water mark rather than live vectors, but because
slots are reused a store only approaches it by *holding* that many at once, not by churning
through them; and raising it later is a `SLOT_BITS` change plus a format decision, not a
redesign.

## Coalescing the free list -- worth doing, not a drop-in

One `VectorID` per hole, 8 bytes each. Bounded by holes rather than by corpus, but the worst
case is not small: deleting half a 20M-vector store costs ~80 MB, and 90% costs ~144 MB. The
filesystem-indexing target makes the bad case likely rather than theoretical -- deleting a
directory frees a long *contiguous* run of slots, which is exactly the shape a flat list stores
worst and an extent list stores best (millions of entries versus one).

The obvious fix is to store extents, `{start, len}`, and coalesce adjacent ones on free. The
reason it is not a drop-in:

**Generations are per-slot.** Today the free list holds ready-to-issue ids -- slot *and* its next
generation, learned from the trailer the open scan was already reading. An extent cannot carry
one generation for the whole run, so an extent-based list has to get the generation from
somewhere else. Three options, none free:

1. **Read it at `put` time.** One extra `pread` on the reuse path to fetch the slot's old
   metadata before overwriting it. Simple and correct; costs a syscall per reused put, which is
   the one thing the current design deliberately avoids.
2. **Keep a separate per-slot generation array.** Removes the read, but it is resident for
   *every* slot rather than every hole -- the id-to-slot map's failure mode in a new hat.
3. **Hybrid.** Extents for long contiguous runs (a deleted directory), individual ids for
   scattered holes (ordinary re-embed churn). Best behaviour, most code, and the split needs a
   heuristic.

Option 1 is probably right if this is ever built: the reuse path is already touching that slot's
chunk, so the read is likely to hit page cache. Not urgent -- 80 MB in the bad case is real but
survivable against a stated tolerance of a few gigabytes -- but worth revisiting before anyone
points the store at a whole filesystem and then deletes a lot of it.

## The metadata write is not atomic, and should be

`put` writes the vector, then writes the 32-byte `VMeta` in one `pwrite`. That write is 32-byte
aligned and never crosses a 4096-byte block, so on real hardware it will not tear -- but that is
a property of the device rather than a guarantee we are owed, and `pfile.writeAt` loops on
partial writes, so a crash mid-loop can leave a torn record.

The blast radius today is bounded, not eliminated. The crc32 covers the vector plus the first 24
metadata bytes, so almost any tear is *detected* -- that slot reads back `Corrupt` instead of
serving wrong data. The exception is a tear confined to `flags`, which is outside the checksum:
there the slot simply reads as not-occupied and the `put` is lost, which is the same outcome as
crashing one instruction earlier. So: containment, not correctness.

The format is already shaped for the fix. `crc32`, `flags` and `_pad` occupy bytes 24..32 -- an
aligned 8-byte word. Writing the 24-byte payload first and then that word as a single 8-byte
commit makes the commit atomic on any hardware worth the name. The cost is a third `pwrite` per
put, which is why it is written down rather than done: `put` is the hot path, and this should
land alongside the other `put` IO work (chunk-at-a-time growth) so the syscall budget is
reasoned about once.

## Deferred, roughly in order

1. **Batched open scan** (priority 1 now), 4 MiB reads. 52x measured, cold; the block-size
   curve is flat above 4 MiB, so this is the whole single-threaded win.
2. **Sidecar index**, back on the list: even batched, a cold open at 20M vectors is ~12 seconds,
   so the scan cannot be the startup path. Must also persist the quantized codes -- recomputing
   them means re-reading 82 GB. Still never inside the data file, still never fatal to lose.
3. **1-bit quantized cache** of the whole DB in memory: scan the codes, then confirm ~100
   candidates on disk. The ~880 bytes of slack per chunk at 768xf32 has room for a 96-byte code,
   so this needs no format change. The measurements make this mandatory, not optional.
4. **Atomic metadata commit** -- split the 32-byte `VMeta` write into a 24-byte payload and an
   8-byte commit word. Land it with the `put` IO work above so the syscall budget is costed once.
5. **RwLock**, once contention is real.
6. **Transactions.**
7. **Coalescing the free list** into extents, if a churned-down corpus makes the flat list's
   worst case (~80 MB at 20M half-deleted) actually bite.
8. **`compact()`**, only if reclaiming space from a churned-down corpus turns out to matter.

**On the sidecar index, twice reversed.** It was first justified by two grounds, then dropped
when both dissolved (reuse needed only an in-memory free list, which is now built and shipped;
the scan looked cheap). The
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
