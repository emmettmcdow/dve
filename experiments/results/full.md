# scan vs. random reads — `full`

- Apple M5, 24 GB RAM, macOS 26.6.2
- warm.dat 4.00 GiB (fits in RAM, read through cache)
- cold.dat 32.00 GiB (exceeds RAM, read with F_NOCACHE)
- random reads per run: 100,000
- generated 2026-09-10 by `./run.sh`

## Sequential full-file scan

| block | warm ms | warm MiB/s | cold ms | cold MiB/s | cold scan of 81.9 GB |
|---|---|---|---|---|---|
| 4 KiB | 835.1 | 4,905 | 238,264.0 | 138 | 9.5 min |
| 16 KiB | 532.9 | 7,687 | 64,470.0 | 508 | 2.6 min |
| 128 KiB | 344.9 | 11,876 | 13,297.1 | 2,464 | 31.7 s |
| 1 MiB | 333.3 | 12,291 | 5,946.6 | 5,510 | 14.2 s |
| 2 MiB | — | — | 5,133.1 | 6,384 | 12.2 s |
| 4 MiB | 337.8 | 12,126 | 4,937.7 | 6,636 | 11.8 s |
| 8 MiB | — | — | 4,895.8 | 6,693 | 11.7 s |
| 16 MiB | 490.4 | 8,353 | 5,170.9 | 6,337 | 12.3 s |
| 32 MiB | — | — | 4,939.7 | 6,634 | 11.8 s |
| 64 MiB | 486.6 | 8,418 | 4,944.7 | 6,627 | 11.8 s |

Fastest cold block: **8 MiB** at 6,693 MiB/s (+/- 0.0%), **11.7 s** for the 20M-vector corpus, 1.21x the 1 MiB block.

## Strided scan (32 B every 4096 B — vstore's open scan)

| file | ms | ns/chunk | vs. full seq @4 KiB |
|---|---|---|---|
| warm | 799.6 | 763 | 0.96x |
| cold | 254,304.2 | 30,315 | 1.07x |

## Random 4 KiB reads

| file | ms | ns/read | IOPS |
|---|---|---|---|
| warm | 133.9 | 1,339 | 746,861 |
| cold | 6,374.3 | 63,743 | 15,688 |

## Crossover — the number this experiment exists for

**warm**: a full scan of 1,048,576 vectors takes 333 ms (best block 1 MiB); one random read costs 1,339 ns.
→ an index must probe fewer than **248,902** vectors (23.74% of the corpus) to beat scanning everything.

**cold**: a full scan of 8,388,608 vectors takes 4,896 ms (best block 8 MiB); one random read costs 63,743 ns.
→ an index must probe fewer than **76,806** vectors (0.92% of the corpus) to beat scanning everything.

