# storebench

A/B measurement of the two vector stores: `src/vec_storage.zig` (**v1**, RAM-resident, an
array of vectors saved whole) against `src/vstore.zig` (**v2**, disk-resident, page-aligned
chunks). It exists because the cutover in `vector.zig` replaced one with the other and the
question "how much did that cost?" deserved a number rather than an intuition.

Unlike the C benchmarks in the parent directory, this one imports both stores and drives their
real APIs. It measures *these two implementations*, not an access pattern, so its results age
with the code in a way `../results/full.md` does not.

```sh
zig build storebench -Doptimize=ReleaseFast
./zig-out/bin/storebench --n 100000 --no-persist
./zig-out/bin/storebench --n 20000                 # includes the per-document persist
./zig-out/bin/storebench --help
```

## What it measures

The corpus is synthetic: 768-dimensional unit vectors drawn around 64 centroids. Real
embeddings cluster, and uniformly random unit vectors in 768 dimensions are so nearly
orthogonal that a 0.5 similarity threshold would reject all of them and `search` would never
touch its priority queue. Both stores are handed a byte-identical corpus from the same seed.

Embedding is deliberately excluded. It costs ~20 ms per sentence, which would bury a storage
difference measured in microseconds.

| phase | what it times |
|---|---|
| `put` | the insert calls alone, every persist call excluded |
| `persist` | `save` (v1) or `flush` (v2), summed over documents |
| `open` | `init` + `load` (v1) or `init` with its trailer scan (v2) |
| `search` | one query against the whole store, averaged |

## The comparison is not symmetric

Reading the `put` and `persist` columns as a like-for-like race would flatter v1:

- **v1's `put` is a memory write.** Nothing reaches disk until `save`, which serializes the
  entire capacity -- every slot, live or not -- and returns once the bytes are in the page
  cache. It does not fsync.
- **v2's `put` is two `pwrite`s** and is durable to the page cache on return. `flush` is an
  `F_FULLFSYNC` barrier, which waits for the drive to commit its own write cache.

So v1-`save` against v2-`flush` compares *written* against *committed*. `--no-persist` drops
both, which isolates the put paths and is the fairest single number.

`--persist-every` matters more than it looks. v1's `save` cost scales with capacity rather than
with what changed, so persisting once per document makes ingest quadratic in the corpus size:
measurable at 20k vectors, hours at 100k. That is not an artifact of the harness -- it is what
`embedText` does on every call.

## Cold vs. warm

Everything above is warm: the file is in the page cache by the time the read phases run, which
flatters v2 by ~40x on exactly the pattern that matters (see `../results/full.md`). `--phase`
runs one phase per process so an external script can `sudo purge` between them:

```sh
./zig-out/bin/storebench --n 200000 --phase ingest --keep
sudo purge
./zig-out/bin/storebench --phase open --reuse
```

`--reuse` is what makes that work: without it every phase rebuilds the corpus first and warms
the page cache doing so, which is precisely what the purge was for.

Memory is a process high-water mark, so the `peak rss` column only separates the two stores
when each runs in its own process: `--store v1` and `--store v2`.
