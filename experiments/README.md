# experiments

Small, permanent, repeatable measurements for design decisions in `dve`. Each experiment is
plain C driven by [hyperfine](https://github.com/sharkdp/hyperfine), deliberately doing the
minimum work that produces the access pattern under test. Nothing here imports `vstore` -- we
measure *patterns*, not formats, so the results stay valid as the on-disk layout changes.

## scan vs. random reads

**Question.** `dve` targets indexing an entire filesystem (~20M vectors). Search must either
scan everything or probe an index. The number that settles which is:

> How many random reads cost the same as one full sequential scan?

An index is worth building only if it probes fewer candidates than that. This experiment
measures it directly instead of extrapolating.

**Why it exists.** Every earlier performance claim about `vstore` rested on a single warm
measurement of a 256 MiB file that fit entirely in RAM, multiplied out to 20M vectors. That
extrapolation crosses a regime boundary it has no right to cross: at filesystem scale the file
is larger than RAM, so nothing caches and every read hits the device.

### Running it

```sh
brew install hyperfine     # prerequisite
make
./run.sh --smoke           # 64 MiB + 256 MiB, a few seconds; checks the plumbing
./run.sh                   # 4 GiB + 32 GiB, the real thing
./run.sh --purge           # additionally `sudo purge` before each run
make clean-data            # reclaim the 36 GiB
```

Results land in `results/<tag>.md`, raw hyperfine JSON in `results/raw/`. Data files and
binaries are gitignored; the result markdown is committed.

### Design notes

**Two files, two regimes.** `warm.dat` (4 GiB) fits in 24 GB of RAM and is read through the
cache -- a true warm number. `cold.dat` (32 GiB) exceeds RAM *and* is read with `F_NOCACHE`.
Both, because either alone is defeatable: `F_NOCACHE` bypasses the page cache but can still be
served from the SSD's own DRAM, and a file merely larger than RAM still keeps its tail cached
between runs. Random reads especially -- 100k x 4 KiB touches only 400 MiB, which would sit in
cache across runs without `--nocache`.

**Measured loops are straight-line.** `common.h` holds setup only; every timed loop is inline
in its own `main` with no calls but `read_exact`, whose overhead is a compare against a ~1 us
syscall. A `volatile` sink accumulates one byte per read so nothing can be elided. Buffers are
`posix_memalign`'d once before timing. `xorshift64` with a fixed seed gives identical offsets
every run. QoS is set to `USER_INTERACTIVE` to prefer performance cores. Single-threaded on
purpose -- concurrency is a separate question and would confound this one.

**Run lengths.** hyperfine times the whole process, so a run must dwarf ~4 ms of startup. This
is why `rand` does 100,000 reads and divides, rather than measuring `log2(20M) ~ 24` reads
directly: 24 reads is ~2 ms and startup would swamp it.

### Sanity checks

The harness is wrong, not the hardware, if any of these fail on a **full** run:

- `warm` sequential throughput should far exceed `cold` (RAM vs. flash).
- `cold` random reads should be far slower than ~2 us. Anything near 400k IOPS means the reads
  are being cached and the file is too small.
- `cold` `stride` should land close to `cold` `seq` at 4 KiB. At 768xf32 each 32-byte trailer
  sits in its own 4 KiB block, so reading every trailer touches every block in the file. If
  `stride` comes out much *faster*, caching is interfering.

The smoke run cannot satisfy the first two -- its files are far smaller than RAM by design.
It exists to prove the plumbing, and it does confirm the third (1.02x).

## Binaries

| binary | pattern | usage |
|---|---|---|
| `mkdata` | writes a test file; skips if already the right size | `./mkdata <file> <bytes>` |
| `seq` | sequential preads, whole file | `./seq <file> <block> [--nocache]` |
| `stride` | `span` bytes at the tail of every `period` bytes | `./stride <file> <period> <span> [--nocache]` |
| `rand` | K random block-aligned preads | `./rand <file> <block> <count> [--nocache]` |

Each prints a one-line summary including the sink value, so a run that got optimised away or
short-read is visible rather than silent.
