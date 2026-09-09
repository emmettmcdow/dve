#!/usr/bin/env python3
"""Turn hyperfine's JSON into the numbers the experiment exists to answer."""
import json, os, subprocess, sys, datetime

tag, warm_sz, cold_sz, rand_n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
MIB = 1024 * 1024


def load(name):
    p = f"results/raw/{tag}_{name}.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["results"][0]
    return r["mean"], r["stddev"] or 0.0


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

w("## Sequential full-file scan\n")
w("| block | warm ms | warm MiB/s | cold ms | cold MiB/s |")
w("|---|---|---|---|---|")
for bs in (4096, 16384, 131072, 1048576):
    ww, cc = load(f"warm_seq_{bs}"), load(f"cold_seq_{bs}")
    if not ww or not cc:
        continue
    w(f"| {bs//1024} KiB | {ww[0]*1e3:.1f} | {warm_sz/MIB/ww[0]:,.0f} "
      f"| {cc[0]*1e3:.1f} | {cold_sz/MIB/cc[0]:,.0f} |")

w("\n## Strided scan (32 B every 4096 B — vstore's open scan)\n")
w("| file | ms | ns/chunk | vs. full seq @4 KiB |")
w("|---|---|---|---|")
for lbl, sz in (("warm", warm_sz), ("cold", cold_sz)):
    s, q = load(f"{lbl}_stride_4096_32"), load(f"{lbl}_seq_4096")
    if not s:
        continue
    n = sz // 4096
    ratio = f"{s[0]/q[0]:.2f}x" if q else "-"
    w(f"| {lbl} | {s[0]*1e3:.1f} | {s[0]/n*1e9:.0f} | {ratio} |")

w("\n## Random 4 KiB reads\n")
w("| file | ms | ns/read | IOPS |")
w("|---|---|---|---|")
for lbl in ("warm", "cold"):
    r = load(f"{lbl}_rand_4096")
    if r:
        w(f"| {lbl} | {r[0]*1e3:.1f} | {r[0]/rand_n*1e9:,.0f} | {rand_n/r[0]:,.0f} |")

w("\n## Crossover — the number this experiment exists for\n")
for lbl, sz in (("warm", warm_sz), ("cold", cold_sz)):
    r = load(f"{lbl}_rand_4096")
    seqs = [(load(f"{lbl}_seq_{b}"), b) for b in (4096, 16384, 131072, 1048576)]
    seqs = [(m, b) for m, b in seqs if m]
    if not r or not seqs:
        continue
    best, bs = min(seqs, key=lambda t: t[0][0])
    per_read = r[0] / rand_n
    crossover = best[0] / per_read
    vecs = sz // 4096
    w(f"**{lbl}**: a full scan of {vecs:,} vectors takes {best[0]*1e3:.0f} ms "
      f"(best block {bs//1024} KiB); one random read costs {per_read*1e9:,.0f} ns.")
    w(f"→ an index must probe fewer than **{crossover:,.0f}** vectors "
      f"({crossover/vecs*100:.2f}% of the corpus) to beat scanning everything.\n")

print("\n".join(out))
