#!/usr/bin/env python3
"""Figure 1: trajectory (left) + noise robustness (right), single-column width."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Categorical color pair for cohort comparison (colorblind-safe blue/orange)
COL_ADNI1 = "#2a78d6"
COL_ADNI2 = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

# -- Data (from trajectory_adni1.py / trajectory_adni2.py and noise_adni1.py / noise_adni2.py outputs) --
rounds = [0, 1, 2, 3, 4]
traj_adni1 = [0.828, 0.944, 0.952, 0.952, 0.952]
traj_adni2 = [0.832, 0.880, 0.880, 0.880, 0.880]

noise_levels = [0, 10, 20]
noise_adni1 = [0.948, 0.948, 0.924]
noise_adni2 = [0.884, 0.876, 0.872]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.8,
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.3), facecolor=SURFACE)

LW = 1.6
MS = 5.5

def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(length=3, width=0.8, colors=INK_SECONDARY)

# -- Panel (a): trajectory ----------------------------------------------------
style_axis(ax1)
ax1.plot(rounds, traj_adni1, color=COL_ADNI1, linewidth=LW, linestyle="-",
          marker="o", markersize=MS, markeredgecolor=SURFACE, markeredgewidth=1.0,
          label="ADNI1", zorder=3)
ax1.plot(rounds, traj_adni2, color=COL_ADNI2, linewidth=LW, linestyle="--",
          marker="s", markersize=MS, markeredgecolor=SURFACE, markeredgewidth=1.0,
          label="ADNI2", zorder=3)
ax1.set_xlabel("Feedback round")
ax1.set_ylabel("P@10")
ax1.set_xticks(rounds)
ax1.set_ylim(0.78, 1.0)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax1.set_title("(a) Convergence", fontsize=8.5, color=INK_PRIMARY, loc="left")
ax1.annotate(f"{traj_adni1[-1]:.3f}", (rounds[-1], traj_adni1[-1]),
             xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)
ax1.annotate(f"{traj_adni2[-1]:.3f}", (rounds[-1], traj_adni2[-1]),
             xytext=(4, -10), textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)

# -- Panel (b): noise robustness ---------------------------------------------
style_axis(ax2)
ax2.plot(noise_levels, noise_adni1, color=COL_ADNI1, linewidth=LW, linestyle="-",
          marker="o", markersize=MS, markeredgecolor=SURFACE, markeredgewidth=1.0,
          label="ADNI1", zorder=3)
ax2.plot(noise_levels, noise_adni2, color=COL_ADNI2, linewidth=LW, linestyle="--",
          marker="s", markersize=MS, markeredgecolor=SURFACE, markeredgewidth=1.0,
          label="ADNI2", zorder=3)
ax2.set_xlabel("Feedback noise level")
ax2.set_xticks(noise_levels)
ax2.set_xticklabels(["0%", "10%", "20%"])
ax2.set_ylim(0.78, 1.0)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax2.set_title("(b) Noise robustness", fontsize=8.5, color=INK_PRIMARY, loc="left")
ax2.annotate(f"{noise_adni1[-1]:.3f}", (noise_levels[-1], noise_adni1[-1]),
             xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)
ax2.annotate(f"{noise_adni2[-1]:.3f}", (noise_levels[-1], noise_adni2[-1]),
             xytext=(4, -10), textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)

# -- Shared legend -------------------------------------------------------------
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.06), fontsize=8.5, handlelength=2.2,
           labelcolor=INK_PRIMARY)

fig.tight_layout(rect=[0, 0, 1, 0.90])

pdf_path = os.path.join(OUT_DIR, "fig1_trajectory_noise.pdf")
png_path = os.path.join(OUT_DIR, "fig1_trajectory_noise.png")
fig.savefig(pdf_path, dpi=300, facecolor=SURFACE, bbox_inches="tight")
fig.savefig(png_path, dpi=300, facecolor=SURFACE, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
