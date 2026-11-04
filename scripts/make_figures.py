"""Figures for the paper and this README, from an evaluated run directory (results.csv, split.csv, per_user.pkl, target_sensitive.pkl).
  privacy_utility: leakage of sensitive products (x) against category HR@10 and semantic similarity@10 (y); an arrow joins
                   each obfuscator without and with the deobfuscator.
  by_target:       utility for users whose next purchase is sensitive vs nonsensitive, with 95% bootstrap intervals.
Colours: BERT = blue, categorical = orange (validated as a pair); reference configurations in neutral gray. Hatched / hollow
marks = without the deobfuscator, so the encoding does not rely on colour. Usage: python scripts/make_figures.py OUT_DIR [RUN_DIR]"""
import sys, os, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]; J = sys.argv[2] if len(sys.argv) > 2 else "runs/main"
os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(f"{J}/results.csv").set_index("scheme"); PER = pickle.load(open(f"{J}/per_user.pkl", "rb"))
split = pd.read_csv(f"{J}/split.csv")
BLUE, ORANGE, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#7a7974", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})

def privacy_utility():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, m, lab in [(axs[0], "cat_hr", "Category HR@10"), (axs[1], "sem_sim", "Semantic similarity@10")]:
        y = lambda k: R.loc[k, m]; x = lambda k: 100 * R.loc[k, "PL_b"]
        for a, b, c, name in [("Categorical obf. only (R')", "Categorical obf. + deobf. (R'')", ORANGE, "Categorical"),
                              ("BERT obf. only (R')", "BERT obf. + deobf. (R'')", BLUE, "BERT")]:
            ax.annotate("", xy=(x(b), y(b)), xytext=(x(a), y(a)), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, shrinkA=5, shrinkB=5))
            ax.scatter(x(a), y(a), s=46, facecolor="white", edgecolor=c, linewidth=1.8, zorder=3)
            ax.scatter(x(b), y(b), s=46, color=c, edgecolor="white", linewidth=1.2, zorder=4)
            ax.annotate(f"{name}\n+ deobf.", (x(b), y(b)), xytext=(7, -2), textcoords="offset points", fontsize=7, color=INK, va="center")
            ax.annotate(f"{name} only", (x(a), y(a)), xytext=(7, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
        for k, name, dx, ha in [("Baseline (R)", "Baseline\n(full history)", -7, "right"), ("Only local", "Only local\n(Llama-3.2-1B)", 7, "left")]:
            ax.scatter(x(k), y(k), s=46, marker="D", color=GRAY, edgecolor="white", linewidth=1.2, zorder=4)
            ax.annotate(name, (x(k), y(k)), xytext=(dx, 0), textcoords="offset points", fontsize=7, color=INK2, va="center", ha=ha)
        for n in ["Qwen3.5-2B", "Qwen3.5-4B"]:
            k = f"Only local [{n}]"; ax.scatter(x(k), y(k), s=22, marker="D", facecolor="white", edgecolor=GRAY, linewidth=1.1, zorder=3)
            ax.annotate(n, (x(k), y(k)), xytext=(7, 0), textcoords="offset points", fontsize=6.5, color=INK2, va="center")
        ax.set_xlim(-8, 112); ax.set_xlabel("Sensitive products sent to the server, PL$_b$ (%)  ← more private")
        ax.set_ylabel(lab + "  (higher is better)")
    fig.tight_layout(w_pad=2.5)
    for ext in ["pdf", "png"]: fig.savefig(f"{OUT}/privacy_utility.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

SCH = [("Baseline (R)", "Baseline", GRAY, False), ("Categorical obf. only (R')", "Categorical only", ORANGE, True),
       ("Categorical obf. + deobf. (R'')", "Categorical + deobf.", ORANGE, False), ("BERT obf. only (R')", "BERT only", BLUE, True),
       ("BERT obf. + deobf. (R'')", "BERT + deobf.", BLUE, False)]
def by_target():
    sp = split.set_index(["target", "scheme"])
    n_s = int(sp.loc[("sensitive", "Baseline (R)"), "n"]); n_n = int(sp.loc[("nonsensitive", "Baseline (R)"), "n"])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    groups = [("sensitive", f"Next purchase sensitive\n(n = {n_s})"), ("nonsensitive", f"Next purchase nonsensitive\n(n = {n_n})")]
    mask = pickle.load(open(f"{J}/target_sensitive.pkl", "rb"))
    rng = np.random.default_rng(0)
    for ax, m, lab in [(axs[0], "cat_hr", "Category HR@10"), (axs[1], "sem_sim", "Semantic similarity@10")]:
        w = 0.15
        for gi, (g, gl) in enumerate(groups):
            sel = mask if g == "sensitive" else ~mask
            for si, (k, name, c, hollow) in enumerate(SCH):
                v = PER[k][m].to_numpy()[sel]; bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
                lo, hi = np.percentile(bs, [2.5, 97.5]); xpos = gi + (si - 2) * (w + 0.02)
                ax.bar(xpos, v.mean(), w, color="white" if hollow else c, edgecolor=c, linewidth=1.2, hatch="////" if hollow else None,
                       label=name if gi == 0 else None, zorder=2)
                ax.errorbar(xpos, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], color=INK, lw=0.9, capsize=1.8, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels([gl for _, gl in groups]); ax.grid(axis="x", visible=False)
        ax.set_ylabel(lab); ax.set_ylim(0, (0.85 if m == "cat_hr" else 0.42))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.07), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(w_pad=2.5)
    for ext in ["pdf", "png"]: fig.savefig(f"{OUT}/by_target.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

privacy_utility(); by_target(); print("figures written to", OUT)
