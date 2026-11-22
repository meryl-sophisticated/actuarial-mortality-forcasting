"""Draw the two figures in figures/ from the files in results/.
Usage: python make_figures.py"""
import json, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = sys.argv[1] if len(sys.argv) > 1 else "."
SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TXT, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.edgecolor": GRID, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
def clean(ax, grid_axis="y"):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)

# 1. mortality: single runs vs ensemble vs Lee-Carter
r = json.load(open(f"{root}/results/mortality_results.json"))
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=False)
for ax, sex in zip(axes, ["Male", "Female"]):
    lc = r[sex]["Lee-Carter"]["out_sample"]["mse_mx_1e4"]
    ax.axhline(lc, color=ORANGE, lw=2); ax.text(1.62, lc, "Lee-Carter", color=TXT2, va="bottom", ha="right", fontsize=9)
    for i, k in enumerate(["GRU", "LSTM"]):
        runs = [x["out_sample"]["mse_mx_1e4"] for x in r[sex][k]["runs"]]
        ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), runs, s=40, color=SURF, edgecolor=BLUE, lw=1.6, zorder=3,
                   label="single run (5 seeds)" if i == 0 else None)
        e = r[sex][k]["ensemble_out_sample"]["mse_mx_1e4"]
        ax.scatter([i + 0.32], [e], s=90, marker="D", color=BLUE, edgecolor=SURF, lw=2, zorder=4,
                   label="5-seed ensemble" if i == 0 else None)
        ax.text(i + 0.45, e, f"{e:.2f}", va="center", fontsize=9, color=TXT)
    ax.set_xticks([0.1, 1.1], ["GRU", "LSTM"]); ax.set_xlim(-0.4, 1.7); ax.set_ylim(0, None)
    ax.set_title(sex, loc="left", fontsize=10.5); clean(ax)
axes[0].set_ylabel("Out-of-sample MSE of m_x, 2000–2016 (×10⁻⁴)")
axes[1].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("Mortality: single networks are unstable; ensembles beat Lee-Carter", x=0.02, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{root}/figures/mortality_ensemble_vs_lee_carter.png", dpi=160); plt.close(fig)

# 2. mortality: male log rates in 2013
d = pd.read_csv(f"{root}/results/mortality_forecasts_2000_2016.csv")
d = d[(d.Gender == "Male") & (d.Year == 2013)]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
obs = d[d.Model == "Lee-Carter"]
ax.scatter(obs.Age, obs.logmx_observed, s=14, color=GRAY, label="Observed", zorder=2)
for name, col, lab in [("Lee-Carter", ORANGE, "Lee-Carter"), ("GRU ensemble", BLUE, "GRU ensemble")]:
    s = d[d.Model == name]; ax.plot(s.Age, s.logmx_forecast, color=col, lw=2, label=lab, zorder=3)
ax.set_xlabel("Age"); ax.set_ylabel("log mortality rate"); clean(ax)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Swiss males, 2013: 14-year-ahead forecasts vs. observed", loc="left", fontsize=11, pad=10)
fig.tight_layout(); fig.savefig(f"{root}/figures/mortality_male_2013.png", dpi=160); plt.close(fig)
print("ok")
