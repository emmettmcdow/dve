# scan vs. random reads — `full`

- Apple M5, 24 GB RAM, macOS 26.6.2
- warm.dat 4.00 GiB (fits in RAM, read through cache)
- cold.dat 32.00 GiB (exceeds RAM, read with F_NOCACHE)
- random reads per run: 100,000
- generated 2026-09-08 by `./run.sh`

## Sequential full-file scan

| block | warm ms | warm MiB/s | cold ms | cold MiB/s |
|---|---|---|---|---|
| 4 KiB | 895.6 | 4,573 | 242045.8 | 135 |
| 16 KiB | 641.0 | 6,390 | 64947.1 | 505 |
| 128 KiB | 413.3 | 9,910 | 13244.9 | 2,474 |
| 1024 KiB | 373.1 | 10,978 | 5749.5 | 5,699 |

## Strided scan (32 B every 4096 B — vstore's open scan)

| file | ms | ns/chunk | vs. full seq @4 KiB |
|---|---|---|---|
| warm | 838.3 | 799 | 0.94x |
| cold | 256198.2 | 30541 | 1.06x |

## Random 4 KiB reads

| file | ms | ns/read | IOPS |
|---|---|---|---|
| warm | 163.3 | 1,633 | 612,308 |
| cold | 6408.2 | 64,082 | 15,605 |

## Crossover — the number this experiment exists for

**warm**: a full scan of 1,048,576 vectors takes 373 ms (best block 1024 KiB); one random read costs 1,633 ns.
→ an index must probe fewer than **228,457** vectors (21.79% of the corpus) to beat scanning everything.

**cold**: a full scan of 8,388,608 vectors takes 5749 ms (best block 1024 KiB); one random read costs 64,082 ns.
→ an index must probe fewer than **89,720** vectors (1.07% of the corpus) to beat scanning everything.

