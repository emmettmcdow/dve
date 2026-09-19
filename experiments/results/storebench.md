# vec_storage.zig vs vstore.zig

Apple M5, 24 GB RAM, macOS 15.6 (Darwin 25.6.0), zig 0.15.2, `-Doptimize=ReleaseFast`.
768-dimensional f32 vectors drawn around 64 centroids, identical corpus to both stores.
100 queries at a 0.5 threshold, ~50 hits per query. Harness: `experiments/storebench`.

**Every number here is warm.** The file is in the page cache by the time the read phases run,
which flatters the disk-resident store by up to 40x on exactly the pattern that matters --
see `full.md`. A cold run needs `--reuse` and `sudo purge` between phases and has not been done.

## Ingest, open and search

`--no-persist`: no `save` and no `flush`, which isolates the insert path from the durability
path. v1's puts are memory writes; v2's are two `pwrite`s each, durable to the page cache.

|                | 20k       | 100k       | 500k          |
|----------------|-----------|------------|---------------|
| **put** v1     | 80,431/s  | 33,091/s   | 8,224/s       |
| **put** v2     | 41,400/s  | 41,500/s   | 37,352/s      |
|                | *v1 1.9x* | *v2 1.25x* | *v2 **4.5x*** |
| **open** v1    | 28.6 ms   | 139.3 ms   | 922.4 ms      |
| **open** v2    | 6.3 ms    | 35.2 ms    | 319.0 ms      |
|                | *v2 4.6x* | *v2 4.0x*  | *v2 2.9x*     |
| **search** v1  | 12.04 ms  | 59.89 ms   | 302.91 ms     |
| **search** v2  | 8.00 ms   | 41.50 ms   | 254.15 ms     |
|                | *v2 1.5x* | *v2 1.4x*  | *v2 1.2x*     |
| **on disk** v1 | 96.8 MB   | 387.1 MB   | 1.5 GB        |
| **on disk** v2 | 78.1 MB   | 390.6 MB   | 1.9 GB        |

**The expected result did not happen.** The prior was that a disk-resident store would lose
badly to one that keeps every vector in RAM. It loses on put only at the smallest size, and by
500k vectors it wins every column except disk.

**v1's put rate falls 10x across the sweep while v2's is flat.** v2's `put` is two `pwrite`s
into a file it extends a chunk at a time -- the same work per vector at any size. v1 doubles a
`[]@Vector(768, f32)`, so every insert past a power of two reallocates, copies and `@memset`s
an array that is 2 GB by the end, and every insert touches a resident array too large to cache.
The two effects are not separated here; the shape of the curve is what matters.

**The disk column is not the 24% page-alignment tax it looks like.** v1's file is sized by
capacity, not by live vectors, so it is 1.24x oversized at 20k (capacity 32,768 for 20,000
vectors) and exactly right at 100k. Only the 500k row, where both are near a power of two,
shows the real ratio: 1.26x, which is the ~24% slack `vec_storage2.md` accounts for.

**Memory is not separated.** `maxrss` is a process high-water mark and both stores run in one
process, so the rss column is the max of the two. Splitting it needs `--store v1` and
`--store v2` in separate processes and has not been done.

## Persisting every document, which is what `embedText` does

N=20,000 in 1,000 documents of 20 sentences, `save`/`flush` after each:

| | total | per document |
|---|---:|---:|
| v1 `save` | 5m 14.6s | 314 ms |
| v2 `flush` | 3.77 s | 3.8 ms |
| | **83x** | |

And this understates it, because the two calls do not promise the same thing: `save` returns
once the bytes are in the page cache, `flush` issues `F_FULLFSYNC` and waits for the drive to
commit its own write cache. v1 is being timed on the cheaper guarantee and still loses by 83x.

The cause is that `save` rewrites the entire capacity every call -- every slot, live or not --
so ingest is quadratic in the corpus. 1,000 documents x 32,768 slots x 3,072 bytes is 100 GB
written to store 61 MB of vectors. It is measurable at 20k and takes hours at 100k, which is
why the sweep above disables it. v2 writes only the chunks that changed.

## What the measurement changed

v2's search was **worse** than v1's when `vector.zig` was first cut over, and two defects
rather than one design difference accounted for it. Measured at 100k, 100 queries:

| | ms/query | vs. previous |
|---|---:|---:|
| as cut over | 164.1 | -- |
| batched whole-store reads | 100.1 | 1.64x |
| `storedDotAt` (no by-value copy) | 41.5 | 2.41x |
| | | **3.95x total** |

1. **The 4 MiB batched read had landed in `scan` and nowhere else.** `search`, `vecsForDoc`
   and `validate` still issued one `pread` per chunk, which at 768xf32 is one syscall per
   vector. `ChunkWalker` is now that loop, written once.
2. **`storedDot` takes its operands by value, and zig pads `@Vector(768, f32)` to 4096 bytes.**
   The inner loop was copying 8 KB per candidate -- twice the bytes the scan itself reads -- so
   the scan was bounded by the copies rather than by the data. `storedDotAt` takes pointers.

**v1 still pays the copy that (2) removes from v2**, so the search column above is not a like-
for-like comparison of the two designs; the architectural gap is smaller than the 1.2-1.5x
shown. That was left alone deliberately: v1 is the store being replaced.

## What this does not measure

- **Cold anything.** The regime that decides the design, and the one `full.md` exists for.
- **Memory per store**, for the reason above.
- **Concurrency.** Both stores are single-threaded here; v2 holds one mutex across its whole
  public surface.
- **Churn.** Every run is insert-only, so v1's leaked slots and v2's free list never come up.
