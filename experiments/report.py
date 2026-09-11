#!/usr/bin/env python3
"""Turn hyperfine's JSON into the numbers the experiment exists to answer.

Reads whatever raw results exist for the tag rather than a fixed list, so a
partial `./run.sh --only ...` refreshes its rows and leaves the rest standing.
"""
import glob, json, os, re, subprocess, sys, datetime

tag, warm_sz, cold_sz, rand_n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
MIB = 1024 * 1024

# The design target: 20M vectors at 768xf32, one 4096-byte chunk each.
TARGET_VECS = 20_000_000
TARGET_BYTES = TARGET_VECS * 4096


def load(name):
    """A bench's (mean, stddev), or None if it has not been run.

    An interrupted hyperfine leaves the export file created but empty, so
    unreadable is treated as absent rather than fatal: one killed bench must not
    take the whole report -- and with it every other bench's numbers -- down.
    """
    p = f"results/raw/{tag}_{name}.json"
    if not os.path.exists(p):
        return None
    try:
        r = json.load(open(p))["results"][0]
        return r["mean"], r["stddev"] or 0.0
    except (json.JSONDecodeError, KeyError, IndexError, OSError):
        print(f"warning: ignoring unreadable {p}", file=sys.stderr)
        return None


def blocks():
    """Every sequential block size measured for this tag, either regime."""
    found = set()
    for p in glob.glob(f"results/raw/{tag}_*_seq_*.json"):
        m = re.search(r"_(?:warm|cold)_seq_(\d+)\.json$", p)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def fmt_bs(b):
    return f"{b >> 20} MiB" if b >= MIB and b % MIB == 0 else f"{b >> 10} KiB"


def fmt_dur(s):
    if s >= 90:
        return f"{s/60:.1f} min"
    return f"{s:.1f} s" if s >= 1 else f"{s*1e3:.0f} ms"


def sysinfo():
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "?"
    ram = int(sh("sysctl -n hw.memsize") or 0) / (1 << 30)
    return sh("sysctl -n machdep.cpu.brand_string"), f"{ram:.0f} GB", sh("sw_vers -productVersion")


cpu, ram, osv = sysinfo()
out = []
w = out.append
w(f"# scan vs. random reads — `{tag}`\n")
w(f"- {cpu}, {ram} RAM, macOS {osv}")
w(f"- warm.dat {warm_sz/2**30:.2f} GiB (fits in RAM, read through cache)")
w(f"- cold.dat {cold_sz/2**30:.2f} GiB (exceeds RAM, read with F_NOCACHE)")
w(f"- random reads per run: {rand_n:,}")
w(f"- generated {datetime.date.today().isoformat()} by `./run.sh`"
  + (" --smoke" if tag == "smoke" else "") + "\n")

BS = blocks()
if BS:
    w("## Sequential full-file scan\n")
    w("| block | warm ms | warm MiB/s | cold ms | cold MiB/s | cold scan of 81.9 GB |")
    w("|---|---|---|---|---|---|")
    for bs in BS:
        ww, cc = load(f"warm_seq_{bs}"), load(f"cold_seq_{bs}")
        if not ww and not cc:
            continue
        wm = f"{ww[0]*1e3:,.1f}" if ww else "—"
        wr = f"{warm_sz/MIB/ww[0]:,.0f}" if ww else "—"
        cm = f"{cc[0]*1e3:,.1f}" if cc else "—"
        cr = f"{cold_sz/MIB/cc[0]:,.0f}" if cc else "—"
        proj = fmt_dur(TARGET_BYTES / (cold_sz / cc[0])) if cc else "—"
        w(f"| {fmt_bs(bs)} | {wm} | {wr} | {cm} | {cr} | {proj} |")

    # The last column is the question the block sweep exists to answer: what a
    # cold, single-threaded, whole-corpus pass costs at the design target.
    colds = [(load(f"cold_seq_{b}"), b) for b in BS]
    colds = [(m, b) for m, b in colds if m]
    if len(colds) > 1:
        best, bb = min(colds, key=lambda t: t[0][0])
        base = next((m for m, b in colds if b == MIB), None)
        rate = cold_sz / MIB / best[0]
        line = (f"\nFastest cold block: **{fmt_bs(bb)}** at {rate:,.0f} MiB/s "
                f"(+/- {best[1]/best[0]*100:.1f}%), "
                f"**{fmt_dur(TARGET_BYTES / (cold_sz/best[0]))}** for the 20M-vector corpus")
        if base and bb != MIB:
            line += f", {base[0]/best[0]:.2f}x the 1 MiB block"
        w(line + ".")

if load("warm_stride_4096_32") or load("cold_stride_4096_32"):
    w("\n## Strided scan (32 B every 4096 B — vstore's open scan)\n")
    w("| file | ms | ns/chunk | vs. full seq @4 KiB |")
    w("|---|---|---|---|")
    for lbl, sz in (("warm", warm_sz), ("cold", cold_sz)):
        s, q = load(f"{lbl}_stride_4096_32"), load(f"{lbl}_seq_4096")
        if not s:
            continue
        n = sz // 4096
        ratio = f"{s[0]/q[0]:.2f}x" if q else "—"
        w(f"| {lbl} | {s[0]*1e3:,.1f} | {s[0]/n*1e9:,.0f} | {ratio} |")

if load("warm_rand_4096") or load("cold_rand_4096"):
    w("\n## Random 4 KiB reads\n")
    w("| file | ms | ns/read | IOPS |")
    w("|---|---|---|---|")
    for lbl in ("warm", "cold"):
        r = load(f"{lbl}_rand_4096")
        if r:
            w(f"| {lbl} | {r[0]*1e3:,.1f} | {r[0]/rand_n*1e9:,.0f} | {rand_n/r[0]:,.0f} |")

cross = []
for lbl, sz in (("warm", warm_sz), ("cold", cold_sz)):
    r = load(f"{lbl}_rand_4096")
    seqs = [(load(f"{lbl}_seq_{b}"), b) for b in BS]
    seqs = [(m, b) for m, b in seqs if m]
    if not r or not seqs:
        continue
    best, bs = min(seqs, key=lambda t: t[0][0])
    per_read = r[0] / rand_n
    crossover = best[0] / per_read
    vecs = sz // 4096
    cross.append(f"**{lbl}**: a full scan of {vecs:,} vectors takes {best[0]*1e3:,.0f} ms "
                 f"(best block {fmt_bs(bs)}); one random read costs {per_read*1e9:,.0f} ns.")
    cross.append(f"→ an index must probe fewer than **{crossover:,.0f}** vectors "
                 f"({crossover/vecs*100:.2f}% of the corpus) to beat scanning everything.\n")
if cross:
    w("\n## Crossover — the number this experiment exists for\n")
    out.extend(cross)

print("\n".join(out))
