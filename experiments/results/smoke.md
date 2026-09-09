# scan vs. random reads — `smoke`

- Apple M5, 24 GB RAM, macOS 26.6.2
- warm.dat 0.06 GiB (fits in RAM, read through cache)
- cold.dat 0.25 GiB (exceeds RAM, read with F_NOCACHE)
- random reads per run: 2,000
- generated 2026-09-08 by `./run.sh` --smoke

## Sequential full-file scan

| block | warm ms | warm MiB/s | cold ms | cold MiB/s |
|---|---|---|---|---|
| 4 KiB | 15.6 | 4,104 | 65.7 | 3,895 |
| 16 KiB | 13.0 | 4,912 | 40.7 | 6,295 |
| 128 KiB | 11.0 | 5,802 | 33.2 | 7,709 |
| 1024 KiB | 11.4 | 5,597 | 34.2 | 7,484 |

## Strided scan (32 B every 4096 B — vstore's open scan)

| file | ms | ns/chunk | vs. full seq @4 KiB |
|---|---|---|---|
| warm | 11.9 | 725 | 0.76x |
| cold | 67.4 | 1028 | 1.02x |

## Random 4 KiB reads

| file | ms | ns/read | IOPS |
|---|---|---|---|
| warm | 4.0 | 2,015 | 496,222 |
| cold | 4.6 | 2,285 | 437,574 |

## Crossover — the number this experiment exists for

**warm**: a full scan of 16,384 vectors takes 11 ms (best block 128 KiB); one random read costs 2,015 ns.
→ an index must probe fewer than **5,474** vectors (33.41% of the corpus) to beat scanning everything.

**cold**: a full scan of 65,536 vectors takes 33 ms (best block 128 KiB); one random read costs 2,285 ns.
→ an index must probe fewer than **14,531** vectors (22.17% of the corpus) to beat scanning everything.

