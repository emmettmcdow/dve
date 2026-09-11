# scan vs. random reads — `smoke`

- Apple M5, 24 GB RAM, macOS 26.6.2
- warm.dat 0.06 GiB (fits in RAM, read through cache)
- cold.dat 0.25 GiB (exceeds RAM, read with F_NOCACHE)
- random reads per run: 2,000
- generated 2026-09-10 by `./run.sh` --smoke

## Sequential full-file scan

| block | warm ms | warm MiB/s | cold ms | cold MiB/s | cold scan of 81.9 GB |
|---|---|---|---|---|---|
| 4 KiB | 14.7 | 4,344 | 64.1 | 3,995 | 19.6 s |
| 16 KiB | 13.0 | 4,912 | 40.7 | 6,295 | 12.4 s |
| 128 KiB | 7.4 | 8,593 | 25.0 | 10,237 | 7.6 s |
| 1 MiB | 7.4 | 8,653 | 24.4 | 10,511 | 7.4 s |
| 4 MiB | 10.3 | 6,198 | 25.4 | 10,065 | 7.8 s |

Fastest cold block: **1 MiB** at 10,511 MiB/s (+/- 5.9%), **7.4 s** for the 20M-vector corpus.

## Strided scan (32 B every 4096 B — vstore's open scan)

| file | ms | ns/chunk | vs. full seq @4 KiB |
|---|---|---|---|
| warm | 10.9 | 668 | 0.74x |
| cold | 62.5 | 953 | 0.97x |

## Random 4 KiB reads

| file | ms | ns/read | IOPS |
|---|---|---|---|
| warm | 4.7 | 2,331 | 429,037 |
| cold | 4.4 | 2,184 | 457,824 |

## Crossover — the number this experiment exists for

**warm**: a full scan of 16,384 vectors takes 7 ms (best block 1 MiB); one random read costs 2,331 ns.
→ an index must probe fewer than **3,173** vectors (19.37% of the corpus) to beat scanning everything.

**cold**: a full scan of 65,536 vectors takes 24 ms (best block 1 MiB); one random read costs 2,184 ns.
→ an index must probe fewer than **11,150** vectors (17.01% of the corpus) to beat scanning everything.

