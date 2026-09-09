#!/usr/bin/env bash
# Drives the scan-vs-random experiment. See README.md for what it answers.
set -euo pipefail
cd "$(dirname "$0")"

command -v hyperfine >/dev/null || { echo "need hyperfine: brew install hyperfine" >&2; exit 1; }
[ -x ./seq ] || { echo "binaries missing, run: make" >&2; exit 1; }

SMOKE=0; PURGE=""
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --purge) PURGE="--prepare 'sudo purge'" ;;
    *) echo "usage: $0 [--smoke] [--purge]" >&2; exit 2 ;;
  esac
done

if [ "$SMOKE" = 1 ]; then
  WARM_SZ=$((64 * 1024 * 1024)); COLD_SZ=$((256 * 1024 * 1024))
  RAND_N=2000; WARM_RUNS=3; COLD_RUNS=3; TAG=smoke
else
  WARM_SZ=$((4 * 1024 * 1024 * 1024)); COLD_SZ=$((32 * 1024 * 1024 * 1024))
  RAND_N=100000; WARM_RUNS=10; COLD_RUNS=3; TAG=full
fi

BLOCKS="4096 16384 131072 1048576"
mkdir -p results/raw
rm -f "results/raw/${TAG}_"*.json

echo "== generating data (${WARM_SZ} + ${COLD_SZ} bytes) =="
./mkdata warm.dat "$WARM_SZ"
./mkdata cold.dat "$COLD_SZ"

# warm.dat fits in RAM and is read through the cache: a true warm measurement.
# cold.dat exceeds RAM *and* is read with F_NOCACHE. Belt and braces, because
# either alone is defeatable -- F_NOCACHE can still be served by the SSD's own
# DRAM, and a >RAM file still caches its tail between runs. Random reads in
# particular touch only RAND_N*4KiB, which would sit in cache across runs
# without --nocache.
bench() { # bench <name> <runs> <warmup> <command...>
  local name="$1" runs="$2" warmup="$3"; shift 3
  echo "-- $name"
  eval hyperfine --style basic --warmup "$warmup" \
    --runs "$runs" $PURGE \
    --command-name "$name" \
    --export-json "results/raw/${TAG}_${name}.json" \
    "'$*'"
}

for bs in $BLOCKS; do
  bench "warm_seq_$bs" "$WARM_RUNS" 3 ./seq warm.dat "$bs"
  bench "cold_seq_$bs" "$COLD_RUNS" 0 ./seq cold.dat "$bs" --nocache
done

# The open-scan pattern: one 32-byte trailer per 4096-byte chunk.
bench "warm_stride_4096_32" "$WARM_RUNS" 3 ./stride warm.dat 4096 32
bench "cold_stride_4096_32" "$COLD_RUNS" 0 ./stride cold.dat 4096 32 --nocache

bench "warm_rand_4096" "$WARM_RUNS" 3 ./rand warm.dat 4096 "$RAND_N"
bench "cold_rand_4096" "$COLD_RUNS" 0 ./rand cold.dat 4096 "$RAND_N"  --nocache

python3 report.py "$TAG" "$WARM_SZ" "$COLD_SZ" "$RAND_N" | tee "results/${TAG}.md"
echo
echo "wrote results/${TAG}.md"
