"""Draws every figure in figures/ from results/raw_results.csv and results/items.jsonl."""
import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results") + os.sep
OUT = os.path.join(ROOT, "figures") + os.sep
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, ACC, GRAY, LIGHT, PAPER = "#16181d", "#4f5561", "#e4e6e9", "#b93815", "#9aa1ac", "#d4d7dc", "#ffffff"
fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Segoe UI", "Inter", "Arial") if f in fonts), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.facecolor": PAPER, "figure.facecolor": PAPER, "font.size": 12})
d = pd.read_csv(R + "raw_results.csv")
items = {}
for l in open(R + "items.jsonl", encoding="utf8"):
    j = json.loads(l); items[j["id"]] = j
d["meta"] = d.id.map(lambda i: items[i]["meta"])

def rate(y, f):
    y = np.asarray(y); f = np.asarray(f).astype(bool)
    a, b = ((y == 1) & f).sum(), (y == 1).sum(); c, e = ((y == 0) & f).sum(), (y == 0).sum()
    return (100 * a / b if b else np.nan), (100 * c / e if e else np.nan)
def style(ax, grid="x"):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
def wilson(k, n, z=1.96):
    import math
    p = k / n; dd = 1 + z * z / n; c = (p + z * z / (2 * n)) / dd; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return 100 * (c - h), 100 * (c + h)
def save(fig, name, **kw): fig.savefig(OUT + name, dpi=170, **kw); plt.close(fig)

# ---------- 1. one cut-off, many tests ----------
dd = d[d.run == "deepset_v1_ctx"].copy(); dd["k"] = dd.id.str.split(":").str[-1].astype(int)
suites = [("deepset (held-out half)", dd[dd.k % 2 == 1]), ("NotInject (all safe)", d[d.run == "notinject_v1_ctx"]),
          ("Hidden in documents,\ndocument question", d[d.run == "bipia_doc"]), ("Hidden in documents,\nplain-attack question", d[d.run == "bipia_v1_asmsg"]),
          ("Obfuscated (9 forms)", d[d.run == "obfuscated_v1_ctx"]), ("Six languages", d[d.run == "multilingual_v1_ctx"]), ("Adaptive attacks", d[d.run == "adaptive_v1_ctx"])]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, idx, title in [(axes[0], 0, "Attacks caught (%)"), (axes[1], 1, "Safe prompts wrongly blocked (%)")]:
    for i, (name, g) in enumerate(suites):
        a = rate(g.label.values, g.jev.values >= 0.5)[idx]; b = rate(g.label.values, g.jev.values >= 0.12)[idx]
        if np.isnan(a): continue
        ax.plot([a, b], [i, i], color=GRID, lw=7, solid_capstyle="round", zorder=1)
        ax.scatter(a, i, s=130, color=GRAY, zorder=3, edgecolor="white", linewidth=2)
        ax.scatter(b, i, s=130, color=ACC, zorder=3, edgecolor="white", linewidth=2)
        ax.text(max(a, b) + 3, i, f"{a:.0f} \u2192 {b:.0f}", va="center", fontsize=10, color=INK2)
    ax.set_title(title, fontsize=12.5, loc="left", pad=10); ax.set_xlim(-3, 125); ax.set_xticks([0, 25, 50, 75, 100]); style(ax)
axes[0].set_yticks(range(len(suites))); axes[0].set_yticklabels([s[0] for s in suites]); axes[0].invert_yaxis()
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GRAY, ms=10, label="cut-off 0.5 (the default)"),
                    Line2D([], [], marker="o", ls="", color=ACC, ms=10, label="cut-off 0.12 (chosen on half of deepset)")],
           frameon=False, loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "cutoff.png")

# ---------- 2. calibration, with and without context ----------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
labels = ["said 0\u201320%", "said 20\u201340%", "said 40\u201360%", "said 60\u201380%", "said 80\u2013100%"]
for ax, run, title in [(axes[0], "deepset_v1_ctx", "With the assistant's purpose as context"), (axes[1], "deepset_v1_noctx", "Without context")]:
    g = d[d.run == run]
    for i, (lo, hi) in enumerate([(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]):
        m = g[(g.jev >= lo) & (g.jev < hi)]
        said, act = 100 * m.jev.mean(), 100 * m.label.mean()
        ax.plot([said, act], [i, i], color=GRID, lw=7, solid_capstyle="round", zorder=1)
        ax.scatter(said, i, s=150, color=GRAY, zorder=3, edgecolor="white", linewidth=2); ax.scatter(act, i, s=150, color=ACC, zorder=3, edgecolor="white", linewidth=2)
        ax.text(107, i, f"n={len(m)}", va="center", fontsize=10, color=INK2)
    ax.set_yticks(range(5)); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(-3, 122); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title(title, fontsize=12.5, loc="left", pad=10); style(ax); ax.set_xlabel("probability of attack (%)")
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GRAY, ms=10, label="what Jev said"), Line2D([], [], marker="o", ls="", color=ACC, ms=10, label="share that really were attacks")],
           frameon=False, loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "calibration.png")

# ---------- 3. allow / review / block ----------
test = dd[dd.k % 2 == 1]
bands = [(0.2, 0.8), (0.1, 0.9), (0.05, 0.3), (0.03, 0.2)]
order = [("allow_ok", "#cfd3d9", "Allowed, correct"), ("allow_miss", ACC, "Allowed, but it was an attack"), ("review", "white", "Sent to a human"),
         ("block_wrong", "#e8a48f", "Blocked, but it was safe"), ("block_ok", INK2, "Blocked, correct")]
fig, ax = plt.subplots(figsize=(10.5, 4.3))
for row, (lo, hi) in enumerate(bands):
    s, y = test.jev.values, test.label.values; n = len(test)
    parts = {"allow_ok": ((s < lo) & (y == 0)).sum(), "allow_miss": ((s < lo) & (y == 1)).sum(), "review": ((s >= lo) & (s < hi)).sum(),
             "block_wrong": ((s >= hi) & (y == 0)).sum(), "block_ok": ((s >= hi) & (y == 1)).sum()}
    left = 0
    for key, col, lab in order:
        v = 100 * parts[key] / n
        ax.barh(row, v, left=left, height=0.58, color=col, edgecolor=(INK2 if key == "review" else "white"), linewidth=(1.2 if key == "review" else 1.5),
                hatch=("////" if key == "review" else None), label=lab if row == 0 else None)
        if v >= 4.5: ax.text(left + v / 2, row, f"{v:.0f}", ha="center", va="center", fontsize=10.5, color=("white" if key in ("block_ok", "allow_miss") else INK),
                             bbox=(dict(boxstyle="round,pad=0.2", fc="white", ec="none") if key == "review" else None))
        left += v
    ax.text(101.5, row, f"{parts['allow_miss']} of {int((y == 1).sum())} attacks let through", va="center", fontsize=10, color=(ACC if parts["allow_miss"] > 10 else INK2))
ax.set_yticks(range(4)); ax.set_yticklabels([f"allow < {lo}, block \u2265 {hi}" for lo, hi in bands]); ax.invert_yaxis(); ax.set_xlim(0, 135)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("share of prompts (%)"); style(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.46, -0.2), ncol=3, fontsize=10)
fig.tight_layout(); save(fig, "bands.png")

# ---------- 4. over-defense (NotInject) ----------
n = d[d.run == "notinject_v1_ctx"]
dets = [("jev", "Jev", ACC, 0.5), ("deepseek", "DeepSeek Flash", GRAY, 0.5), ("regex", "Regex rules", LIGHT, 1), ("protectai", "ProtectAI DeBERTa", INK, 0.5)]
fig, ax = plt.subplots(figsize=(9.5, 3.8))
for i, (c, label, col, th) in enumerate(dets):
    k = int((n[c] >= th).sum()); N = len(n); lo, hi = wilson(k, N)
    ax.plot([lo, hi], [i, i], color=col if col != LIGHT else GRAY, lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(100 * k / N, i, s=160, color=col, zorder=3, edgecolor="white" if col != LIGHT else GRAY, linewidth=2)
    ax.text(hi + 2, i, f"{100*k/N:.1f}%  ({k} of {N})", va="center", fontsize=10.5, color=INK2)
ax.set_yticks(range(4)); ax.set_yticklabels([x[1] for x in dets]); ax.invert_yaxis(); ax.set_xlim(-2, 75); ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel("safe prompts wrongly flagged (%), with 95% interval"); style(ax)
fig.tight_layout(); save(fig, "notinject.png")

# ---------- 5. hidden attacks in documents: score strips ----------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
rng = np.random.default_rng(1)
for ax, run, title in [(axes[0], "bipia_doc", "Asked the document question"), (axes[1], "bipia_v1_asmsg", "Asked the plain-attack question")]:
    g = d[d.run == run]
    for xpos, lab, col in [(0, 0, GRAY), (1, 1, ACC)]:
        v = g[g.label == lab].jev.values
        ax.scatter(xpos + rng.uniform(-0.28, 0.28, len(v)), v, s=26, color=col, alpha=0.8, edgecolor="white", linewidth=0.6, zorder=3)
        ax.hlines(np.median(v), xpos - 0.34, xpos + 0.34, color=INK, lw=2.5, zorder=4)
    ax.axhline(0.5, color=INK2, lw=1.2, ls=":"); ax.text(1.62, 0.52, "cut-off 0.5", ha="right", fontsize=9.5, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["150 clean documents", "150 with a hidden attack"]); ax.set_xlim(-0.6, 1.65)
    ax.set_title(title, fontsize=12.5, loc="left", pad=10); style(ax, "y"); ax.spines["bottom"].set_visible(True); ax.tick_params(axis="y", length=3)
axes[0].set_ylabel("Jev's probability of attack"); axes[0].set_ylim(-0.03, 1.03)
fig.legend(handles=[Line2D([], [], color=INK, lw=2.5, label="median")], frameon=False, loc="lower center", fontsize=10.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, "documents.png")

# ---------- 6. obfuscation and languages: two-panel dots, Jev vs ProtectAI ----------
def two_panel(run, rows, labels, name, figsize, xlabelL="Attacks caught (%)", xlabelR="Safe prompts wrongly blocked (%)"):
    g = d[d.run == run]; fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, idx, title in [(axes[0], 0, xlabelL), (axes[1], 1, xlabelR)]:
        for i, key in enumerate(rows):
            x = g[g.variant == key]
            j = rate(x.label.values, x.jev.values >= 0.5)[idx]; p = rate(x.label.values, x.protectai.values >= 0.5)[idx]
            ax.plot([j, p], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
            ax.scatter(p, i, s=110, color=INK, zorder=3, edgecolor="white", linewidth=1.8); ax.scatter(j, i, s=130, color=ACC, zorder=3, edgecolor="white", linewidth=1.8)
        ax.set_title(title, fontsize=12.5, loc="left", pad=10); ax.set_xlim(-3, 106); ax.set_xticks([0, 25, 50, 75, 100]); style(ax)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=ACC, ms=10, label="Jev (cut-off 0.5)"), Line2D([], [], marker="o", ls="", color=INK, ms=9, label="ProtectAI (cut-off 0.5)")],
               frameon=False, loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, name)
two_panel("obfuscated_v1_ctx", ["plain", "base64", "rot13", "leetspeak", "spaced", "zero_width", "homoglyph", "reversed", "html_comment"],
          ["Plain text", "Base64", "ROT13", "Leetspeak", "Spaced letters", "Invisible characters", "Look-alike letters", "Reversed", "HTML comment"], "obfuscation.png", (11.5, 5.2))
two_panel("multilingual_v1_ctx", ["de", "fr", "es", "zh", "ar", "hi"], ["German", "French", "Spanish", "Chinese", "Arabic", "Hindi"], "languages.png", (11.5, 4.2))

# ---------- 7. adaptive attacks: every prompt as a dot ----------
a = d[d.run == "adaptive_v1_ctx"]; tactics = ["operator_claim", "fake_system", "criteria_mimic", "meta_denial", "security_class", "roleplay", "hypothetical", "polite", "translate_wrap", "needle"]
names = {"operator_claim": "Claims to be the operator", "fake_system": "Fake system message", "criteria_mimic": "Mimics Jev's own wording", "meta_denial": "\u201cThis is not an attack\u201d",
         "security_class": "\u201cFor a security course\u201d", "roleplay": "Role-play as \u201cMax\u201d", "hypothetical": "Hypothetical framing", "polite": "Very polite", "translate_wrap": "Wrapped in a translation task", "needle": "Buried in a long text"}
fig, ax = plt.subplots(figsize=(10.5, 5.4))
for i, t in enumerate(tactics):
    x = a[a.variant == t]
    for lab, col, off in [(1, ACC, -0.12), (0, GRAY, 0.12)]:
        v = x[x.label == lab].jev.values
        ax.scatter(v, np.full(len(v), i + off) + rng.uniform(-0.07, 0.07, len(v)), s=60, color=col, alpha=0.9, edgecolor="white", linewidth=0.9, zorder=3)
ax.axvline(0.5, color=INK2, lw=1.2, ls=":"); ax.text(0.51, -0.62, "cut-off 0.5", fontsize=9.5, color=INK2)
ax.set_yticks(range(len(tactics))); ax.set_yticklabels([names[t] for t in tactics]); ax.invert_yaxis(); ax.set_xlim(-0.03, 1.03); ax.set_ylim(len(tactics) - 0.4, -0.9)
ax.set_xlabel("Jev's probability of attack"); style(ax)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=ACC, ms=9, label="attack (6 per tactic)"), Line2D([], [], marker="o", ls="", color=GRAY, ms=9, label="harmless twin with the same framing (6 per tactic, where one makes sense)")],
           frameon=False, loc="lower center", ncol=2, fontsize=10.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "adaptive.png")

# ---------- 8. wording and context on deepset: catch across cut-offs ----------
fig, ax = plt.subplots(figsize=(9.5, 4.6))
cols = {"deepset_v1_ctx": (ACC, "My wording (V1)"), "deepset_v1_noctx": (ACC, "V1, no context"), "deepset_v2_ctx": (INK, "Short wording (V2)"), "deepset_v3_ctx": (GRAY, "\u201cWould it break the rules?\u201d (V3)")}
ths = np.linspace(0.01, 0.99, 99)
for run, (col, lab) in cols.items():
    g = d[d.run == run]; ys = [rate(g.label.values, g.jev.values >= t)[0] for t in ths]
    ax.plot(ths * 100, ys, color=col, lw=2.4, ls="--" if run.endswith("noctx") else "-", label=lab)
ax.axvline(50, color=INK2, lw=1, ls=":"); ax.text(51, 4, "default 0.5", fontsize=9.5, color=INK2)
ax.set_xlabel("cut-off (probability, %)"); ax.set_ylabel("attacks caught (%)"); ax.set_ylim(0, 102); ax.set_xlim(0, 100)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.grid(color=GRID, lw=1); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=10.5, loc="upper right")
fig.tight_layout(); save(fig, "wording.png")

print("charts done:", sorted(os.listdir(OUT)))
