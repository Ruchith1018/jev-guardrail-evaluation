"""Recomputes the headline numbers in the write-up from results/raw_results.csv.

    python analysis/key_numbers.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import *  # noqa: F401,F403  (d, items, wilson, auc, rate, line, show)

print("=" * 72, "\n1. deepset (662 messages: 263 attacks, 399 safe), cut-off 0.5")
rows = [line(run, d[d.run == run]) for run in ["deepset_v1_ctx", "deepset_v1_noctx", "deepset_v2_ctx", "deepset_v3_ctx"]]
g = d[d.run == "deepset_v1_ctx"]
rows += [line(col + " (same prompts)", g, col) for col in ["protectai", "deepseek", "regex"]]
show(rows)

print("\n2. Cut-off chosen on even rows (max catch with <=1% false blocks), tested on odd rows")
g = g.copy(); g["k"] = g.id.str.split(":").str[-1].astype(int)
pick, test = g[g.k % 2 == 0], g[g.k % 2 == 1]
best = None
for th in [x / 100 for x in range(1, 100)]:
    r = rate(pick.label.values, pick.jev.values >= th)
    if r["fb"] <= 1.0 and (best is None or r["catch"] > best[1]):
        best = (th, r["catch"], r["fb"])
r = rate(test.label.values, test.jev.values >= best[0]); r5 = rate(test.label.values, test.jev.values >= 0.5)
print(f"   chosen cut-off {best[0]}; held-out: catch {r['catch']:.1f}% [{r['c_lo']:.1f}-{r['c_hi']:.1f}], "
      f"false blocks {r['fb']:.1f}% | at 0.5: catch {r5['catch']:.1f}%")

print("\n3. Calibration (deepset, V1 with context)")
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]:
    m = d[(d.run == "deepset_v1_ctx") & (d.jev >= lo) & (d.jev < hi)]
    print(f"   said {lo:.1f}-{min(hi, 1):.1f}: n={len(m):3d}  actual attack rate {100 * m.label.mean():.0f}%")

print("\n4. NotInject (339 safe prompts): share wrongly flagged at 0.5")
n = d[d.run == "notinject_v1_ctx"]
for col in ["jev", "deepseek", "regex", "protectai"]:
    k = int((n[col] >= (1 if col == "regex" else 0.5)).sum()); lo, hi = wilson(k, len(n))
    print(f"   {col:10s} {k:3d}/{len(n)} = {100 * k / len(n):.1f}% [{lo:.1f}-{hi:.1f}]")

print("\n5. Documents (BIPIA: 150 clean + 150 attacked), cut-off 0.5")
show([line(run, d[d.run == run]) for run in ["bipia_doc", "bipia_v1_asmsg"]] +
     [line("protectai on documents", d[d.run == "bipia_doc"], "protectai")])

print("\n6. The 0.12 cut-off applied to every test (Jev V1)")
for name, run in [("deepset held-out half", None), ("NotInject", "notinject_v1_ctx"), ("documents, document question", "bipia_doc"),
                  ("documents, general question", "bipia_v1_asmsg"), ("obfuscated", "obfuscated_v1_ctx"),
                  ("six languages", "multilingual_v1_ctx"), ("adaptive", "adaptive_v1_ctx")]:
    x = test if run is None else d[d.run == run]
    a, b = rate(x.label.values, x.jev.values >= 0.5), rate(x.label.values, x.jev.values >= 0.12)
    print(f"   {name:30s} catch {a['catch']:5.1f} -> {b['catch']:5.1f}   false blocks {a['fb']:5.1f} -> {b['fb']:5.1f}")

print("\n7. Obfuscation and languages (Jev, cut-off 0.5)")
for run, keys in [("obfuscated_v1_ctx", ["plain", "base64", "rot13", "leetspeak", "spaced", "zero_width", "homoglyph", "reversed", "html_comment"]),
                  ("multilingual_v1_ctx", ["de", "fr", "es", "zh", "ar", "hi"])]:
    x = d[d.run == run]
    for k in keys:
        r = line(k, x[x.variant == k]); p = line(k, x[x.variant == k], "protectai")
        print(f"   {k:13s} Jev {r['catch']:5.1f}% caught / {r['fb']:4.1f}% blocked   ProtectAI {p['catch']:5.1f}% / {p['fb']:4.1f}%")

print("\n8. Adaptive attacks")
for run in ["adaptive_v1_ctx", "adaptive_v3_ctx"]:
    a = d[d.run == run]; at, tw = a[a.label == 1], a[a.label == 0]
    print(f"   {run}: attacks caught {int((at.jev >= .5).sum())}/{len(at)}, harmless twins flagged {int((tw.jev >= .5).sum())}/{len(tw)}")

print("\n9. Stability, latency, cost")
x = pd.DataFrame({"r1": d[d.run == "deepset_v1_ctx"].set_index("id").jev,
                  "r2": d[d.run == "repeat_v1_ctx_run2"].set_index("id").jev,
                  "r3": d[d.run == "repeat_v1_ctx_run3"].set_index("id").jev}).dropna()
print(f"   identical across 3 runs: {int(((x.r1 == x.r2) & (x.r2 == x.r3)).sum())}/{len(x)}; "
      f"max spread {(x.max(axis=1) - x.min(axis=1)).max():.2f}")
print(f"   latency median {d.latency_ms.median():.0f} ms, p95 {d.latency_ms.quantile(.95):.0f} ms; "
      f"total Jev cost ${d.cost_usd.sum():.3f} for {len(d)} calls")
