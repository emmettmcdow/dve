#!/usr/bin/env bash
# Drives the scan-vs-random experiment. See README.md for what it answers.
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat >&2 <<'USAGE'
usage: ./run.sh [options]

  --smoke              small files (64 MiB / 256 MiB), seconds; checks plumbing
  --purge              `sudo purge` before every timed run
  --fresh              discard this tag's existing raw results before running

Selecting a subset. The report merges whatever raw results exist for the tag, so
a partial run refreshes only what it measures and leaves the rest standing --
which is the point: a cold 4 KiB scan of 32 GiB costs four minutes a run, and
nothing else should have to wait behind it.

  --only LIST          benches to run:  seq,stride,rand     (default: all three)
  --regime LIST        regimes to run:  warm,cold           (default: both)
  --blocks LIST        sequential block sizes; K/M/G suffixes accepted
  --runs N             timed runs per bench (overrides the per-regime default)
  --list               print what this invocation would run, then exit
  -h, --help           this

examples:
  ./run.sh --smoke                                  # plumbing, a few seconds
  ./run.sh --only seq --regime cold --blocks 1M,64M # just the big-block question
  ./run.sh --only rand                              # refresh one row of the report
USAGE
  exit 2
}

# Bytes, accepting a K/M/G suffix so --blocks 64M beats counting zeroes.
parse_size() {
  local s="$1" mult=1
  case "$s" in
    *[Kk]) mult=1024 ; s="${s%?}" ;;
    *[Mm]) mult=1048576 ; s="${s%?}" ;;
    *[Gg]) mult=1073741824 ; s="${s%?}" ;;
  esac
  case "$s" in ''|*[!0-9]*) echo "bad size: $1" >&2; exit 2 ;; esac
  echo $((s * mult))
}

# Accept --opt=value as well as --opt value.
split=()
for a in "$@"; do
  case "$a" in
    --*=*) split+=("${a%%=*}" "${a#*=}") ;;
    *)     split+=("$a") ;;
  esac
done
set -- ${split[@]+"${split[@]}"}

SMOKE=0; PURGE=""; FRESH=0; LIST=0
BENCHES="seq stride rand"; REGIMES="warm cold"; BLOCKS=""; RUNS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)  SMOKE=1 ;;
    --purge)  PURGE="--prepare 'sudo purge'" ;;
    --fresh)  FRESH=1 ;;
    --list)   LIST=1 ;;
    --only)   [ $# -ge 2 ] || usage; BENCHES="${2//,/ }"; shift ;;
    --regime) [ $# -ge 2 ] || usage; REGIMES="${2//,/ }"; shift ;;
    --blocks) [ $# -ge 2 ] || usage; BLOCKS="${2//,/ }";  shift ;;
    --runs)   [ $# -ge 2 ] || usage; RUNS="$2";           shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
  shift
done

for b in $BENCHES; do
  case "$b" in seq|stride|rand) ;; *) echo "unknown bench: $b" >&2; usage ;; esac
done
for r in $REGIMES; do
  case "$r" in warm|cold) ;; *) echo "unknown regime: $r" >&2; usage ;; esac
done

if [ "$SMOKE" = 1 ]; then
  WARM_SZ=$((64 * 1024 * 1024)); COLD_SZ=$((256 * 1024 * 1024))
  RAND_N=2000; WARM_RUNS=3; COLD_RUNS=3; TAG=smoke
  DEFAULT_BLOCKS="4096 131072 1048576 4194304"
else
  WARM_SZ=$((4 * 1024 * 1024 * 1024)); COLD_SZ=$((32 * 1024 * 1024 * 1024))
  RAND_N=100000; WARM_RUNS=10; COLD_RUNS=3; TAG=full
  # Past 1 MiB the warm curve has flattened but the cold one had not, which is
  # why the list keeps climbing: the cold knee is the number the index build
  # cares about, and it was never bracketed from above.
  DEFAULT_BLOCKS="4096 16384 131072 1048576 4194304 16777216 67108864"
fi
[ -n "$BLOCKS" ] || BLOCKS="$DEFAULT_BLOCKS"
[ -z "$RUNS" ] || { WARM_RUNS="$RUNS"; COLD_RUNS="$RUNS"; }

norm=""
for b in $BLOCKS; do
  n="$(parse_size "$b")"
  # seq refuses these too, but only after hyperfine has started and, on a first
  # run, after mkdata has written 32 GiB. Fail before that.
  { [ "$n" -gt 0 ] && [ $((n % 4096)) -eq 0 ]; } ||
    { echo "block must be a nonzero multiple of 4096: $b" >&2; exit 2; }
  norm="$norm $n"
done
BLOCKS="${norm# }"

want() { # want <bench> <regime>
  case " $BENCHES " in *" $1 "*) ;; *) return 1 ;; esac
  case " $REGIMES " in *" $2 "*) ;; *) return 1 ;; esac
}

plan() { # the benches this invocation covers, in run order
  local bs
  for bs in $BLOCKS; do
    want seq warm && echo "warm_seq_$bs"
    want seq cold && echo "cold_seq_$bs"
  done
  want stride warm && echo "warm_stride_4096_32"
  want stride cold && echo "cold_stride_4096_32"
  want rand warm   && echo "warm_rand_4096"
  want rand cold   && echo "cold_rand_4096"
  true
}

if [ "$LIST" = 1 ]; then
  echo "tag: $TAG   warm: $WARM_SZ B   cold: $COLD_SZ B   rand reads: $RAND_N"
  plan
  exit 0
fi

command -v hyperfine >/dev/null || { echo "need hyperfine: brew install hyperfine" >&2; exit 1; }
[ -x ./seq ] || { echo "binaries missing, run: make" >&2; exit 1; }

mkdir -p results/raw

# Raw results accumulate across partial runs, so they have to be comparable.
# File sizes and the random-read count are what make them so; the block list is
# not, since every block writes its own file. A change to the former discards
# the tag's results rather than quietly reporting two geometries in one table.
META="results/raw/${TAG}_meta.json"
GEOM="{\"warm_sz\":$WARM_SZ,\"cold_sz\":$COLD_SZ,\"rand_n\":$RAND_N}"
if [ "$FRESH" = 1 ]; then
  rm -f "results/raw/${TAG}_"*.json
elif [ -f "$META" ] && [ "$(cat "$META")" != "$GEOM" ]; then
  echo "== geometry changed since the last $TAG run; discarding stale raw results =="
  rm -f "results/raw/${TAG}_"*.json
fi
printf '%s\n' "$GEOM" > "$META"

# Generate only what the selected regimes read: an unwanted cold.dat is 32 GiB
# and half an hour.
case " $REGIMES " in *" warm "*)
  echo "== warm.dat ($WARM_SZ bytes) =="; ./mkdata warm.dat "$WARM_SZ" ;;
esac
case " $REGIMES " in *" cold "*)
  echo "== cold.dat ($COLD_SZ bytes) =="; ./mkdata cold.dat "$COLD_SZ" ;;
esac

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
  if want seq warm; then bench "warm_seq_$bs" "$WARM_RUNS" 3 ./seq warm.dat "$bs"; fi
  if want seq cold; then bench "cold_seq_$bs" "$COLD_RUNS" 0 ./seq cold.dat "$bs" --nocache; fi
done

# The open-scan pattern: one 32-byte trailer per 4096-byte chunk.
if want stride warm; then bench "warm_stride_4096_32" "$WARM_RUNS" 3 ./stride warm.dat 4096 32; fi
if want stride cold; then bench "cold_stride_4096_32" "$COLD_RUNS" 0 ./stride cold.dat 4096 32 --nocache; fi

if want rand warm; then bench "warm_rand_4096" "$WARM_RUNS" 3 ./rand warm.dat 4096 "$RAND_N"; fi
if want rand cold; then bench "cold_rand_4096" "$COLD_RUNS" 0 ./rand cold.dat 4096 "$RAND_N" --nocache; fi

# Rendered to a temporary file first. `| tee results/$TAG.md` truncates the
# previous report before the generator has produced a line, so a generator that
# dies takes committed results with it -- which is exactly what happened once.
python3 report.py "$TAG" "$WARM_SZ" "$COLD_SZ" "$RAND_N" > "results/.${TAG}.md.tmp"
mv "results/.${TAG}.md.tmp" "results/${TAG}.md"
cat "results/${TAG}.md"
echo
echo "wrote results/${TAG}.md"
