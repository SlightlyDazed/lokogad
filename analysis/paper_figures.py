#!/usr/bin/env python3
"""
paper_figures.py — English figures for the journal article, from the stored results.

  figconvergence.png    topological search: fitness, errors and valid designs per generation
  fig2_magic_numbers.png  parametric-stage genome vs Jansen's original values
  fig3_pareto_front.png   50 best parametric solutions: search metric and re-evaluated shape error
  fig4_z_layers.png       layer assignment by graph colouring and the kinematic conflict graph
  fig_gait_sequence_3d.png  final mechanism at six crank angles and the full foot path (2-D)

Usage (from the repository root):
    python analysis/paper_figures.py [--out results/paper_figures]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

C_FOOT, C_KNEE, C_POS, C_FIT, C_ORI = "#c0392b", "#2471a3", "#7d3c98", "#1c2833", "#95a5a6"
LAYER_PITCH_MM = 6.6

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white",
})


def panel(ax, letter):
    ax.text(-0.13, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=11)


def fig_convergence(out):
    h = json.load(open(os.path.join(RES, "evolution_history.json")))
    g = np.array([r["gen"] for r in h])
    get = lambda k: np.array([r[k] for r in h])
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
    a = ax[0, 0]; a.plot(g, get("best_fitness"), color=C_FIT, lw=1.8)
    a.set_title(f"Fitness of the best individual (final {get('best_fitness')[-1]:.5f})")
    a.set_ylabel("Fitness F"); panel(a, "a")
    a = ax[0, 1]
    a.plot(g, 100 * get("best_foot"), color=C_FOOT, lw=1.8, label=f"Foot (final {100*get('best_foot')[-1]:.2f}%)")
    a.plot(g, 100 * get("best_knee"), color=C_KNEE, lw=1.8, label=f"Knee (final {100*get('best_knee')[-1]:.2f}%)")
    a.set_title("Trajectory errors of the best individual\n(search metric, original implementation)")
    a.set_ylabel("Error (%)"); a.legend(frameon=False); panel(a, "b")
    a = ax[1, 0]; a.plot(g, 100 * get("best_pos"), color=C_POS, lw=1.8)
    a.set_title(f"Position error of the best individual (final {100*get('best_pos')[-1]:.1f}%)")
    a.set_ylabel("Error (%)"); a.set_xlabel("Generation"); panel(a, "c")
    a = ax[1, 1]; nv = get("n_valid")
    a.fill_between(g, nv, color="#566573", alpha=0.25); a.plot(g, nv, color="#566573", lw=1.4)
    a.set_title("Designs that assemble at all 360 crank angles")
    a.set_ylabel("Valid designs (of 5,000)"); a.set_xlabel("Generation"); a.set_ylim(0, 5000); panel(a, "d")
    fig.tight_layout(); fig.savefig(os.path.join(out, "figconvergence.png"), dpi=300); plt.close(fig)


def fig_genome(out):
    b = json.load(open(os.path.join(RES, "best_mechanism.json")))
    opt, ori, sc = np.array(b["magic_numbers"]), np.array(b["magic_numbers_original"]), np.array(b["scale_vs_original"])
    lab = [str(i + 1) for i in range(13)]
    x = np.arange(13)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    a = ax[0]
    a.bar(x - 0.2, ori / ori.max(), 0.4, color=C_ORI, label="Jansen's original")
    a.bar(x + 0.2, opt / opt.max(), 0.4, color=C_KNEE, label="Optimised (NSGA-II)")
    a.set_xticks(x, lab); a.set_xlabel("Magic Number"); a.set_ylabel("Length / largest length")
    a.set_title("Normalised lengths"); a.legend(frameon=False, fontsize=9); panel(a, "a")
    a = ax[1]
    bars = a.bar(x, sc, color=C_KNEE)
    a.axhline(sc.mean(), color="k", ls="--", lw=1, label=f"Mean ×{sc.mean():.1f}")
    for r, v in zip(bars, sc):
        a.text(r.get_x() + r.get_width() / 2, v + 0.4, f"×{v:.1f}", ha="center", fontsize=7.5)
    a.set_xticks(x, lab); a.set_xlabel("Magic Number"); a.set_ylabel("Optimised / original")
    a.set_title("Scale factor per dimension"); a.legend(frameon=False, fontsize=9); panel(a, "b")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_magic_numbers.png"), dpi=300); plt.close(fig)


def fig_pareto(out):
    import config_jansen as cj
    from shape_fit import evaluate, load_refs
    F = np.load(os.path.join(RES, "top_F_exact.npy"))
    MN = np.load(os.path.join(RES, "top_MN.npy"))
    ref = load_refs()
    sf, sk, pf = [], [], []
    for mn in MN:
        tj = cj.compute_trajectories(np.asarray(mn, dtype=float))
        e = evaluate(tj[cj.POINT_FOOT], tj[cj.POINT_KNEE], ref)
        sf.append(100 * e["foot"]["shape"]); sk.append(100 * e["knee"]["shape"])
        pf.append(100 * (0.6 * e["pos_foot"] + 0.4 * e["pos_knee"]))
    sf, sk, pf = map(np.array, (sf, sk, pf))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    s = a.scatter(100 * F[:, 0], 100 * F[:, 1], c=100 * F[:, 2], cmap="viridis", s=36, edgecolors="k", lw=0.3)
    a.scatter(100 * F[0, 0], 100 * F[0, 1], marker="*", s=220, c="none", edgecolors=C_FOOT, lw=1.4,
              label=f"Lowest foot error ({100*F[0,0]:.2f}%)")
    fig.colorbar(s, ax=a, label="Position error (%)")
    a.set_xlabel("Foot error (%)"); a.set_ylabel("Knee error (%)")
    a.set_title("Search metric (original implementation)"); a.legend(frameon=False, fontsize=8.5, loc="lower left")
    panel(a, "a")
    a = ax[1]
    s = a.scatter(sf, sk, c=pf, cmap="viridis", s=36, edgecolors="k", lw=0.3)
    fig.colorbar(s, ax=a, label="Weighted position error (%)")
    a.set_xlabel("Foot shape error (%)"); a.set_ylabel("Knee shape error (%)")
    a.set_title("Re-evaluated with the shape error (all mirrored)"); panel(a, "b")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_pareto_front.png"), dpi=300); plt.close(fig)
    print(f"  re-evaluated top-50: foot {sf.min():.1f}-{sf.max():.1f}%, knee {sk.min():.1f}-{sk.max():.1f}%")


def fig_layers(out):
    z = json.load(open(os.path.join(RES, "z_layers_fixed.json")))
    L, conf = np.array(z["z_layers"]), z["geo_conflicts"]
    g = json.load(open(os.path.join(RES, "best_genome.json")))
    names = [f"{i}  ({e['n1']}–{e['n2']})" for i, e in enumerate(g["edges"])]
    levels = sorted(set(L.tolist()))
    cmap = plt.cm.viridis(np.linspace(0.08, 0.92, len(levels)))
    col = {lv: cmap[i] for i, lv in enumerate(levels)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    for i, lv in enumerate(L):
        a.barh(i, lv * LAYER_PITCH_MM, color=col[lv], height=0.62)
        a.text(lv * LAYER_PITCH_MM + (0.6 if lv >= 0 else -0.6), i, f"{lv*LAYER_PITCH_MM:+.1f}",
               va="center", ha="left" if lv >= 0 else "right", fontsize=8)
    a.axvline(0, color="k", lw=0.8)
    a.set_yticks(range(len(L)), names, fontsize=8.5); a.invert_yaxis()
    a.set_xlim(-25, 18); a.set_xlabel("Axial offset of the bar (mm)")
    a.set_title(f"Layer per bar ({len(levels)} layers, pitch {LAYER_PITCH_MM} mm)"); panel(a, "a")
    a = ax[1]; a.grid(False); a.axis("off")
    n = len(L)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    P = np.column_stack([np.cos(ang), np.sin(ang)])
    for b1, b2 in conf:
        a.plot(*P[[b1, b2]].T, color=C_FOOT, lw=1.8, zorder=1)
    for i in range(n):
        a.add_patch(plt.Circle(P[i], 0.13, color=col[L[i]], ec="k", lw=0.6, zorder=2))
        a.text(*P[i], str(i), ha="center", va="center", fontsize=9, zorder=3,
               color="white" if L[i] <= levels[len(levels) // 2 - 1] else "black", fontweight="bold")
    a.set_xlim(-1.35, 1.35); a.set_ylim(-1.35, 1.35); a.set_aspect("equal")
    a.set_title(f"Kinematic conflict graph ({len(conf)} conflicting pairs)"); panel(a, "b")
    h = [mpatches.Patch(color=col[lv], label=f"{lv*LAYER_PITCH_MM:+.1f} mm") for lv in levels]
    h.append(plt.Line2D([], [], color=C_FOOT, lw=1.8, label="Conflict"))
    a.legend(handles=h, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, fontsize=8, title="Layer")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_z_layers.png"), dpi=300); plt.close(fig)


def fig_gait_sequence(out, scale=0.1713, band=0.35):
    """Mechanism at six crank angles and the full foot path (simulated layout, prototype scale)."""
    from general_solver import compute_full_trajectory
    from linkage_genome import LinkageGenome
    gd = json.load(open(os.path.join(RES, "best_genome.json")))
    g = LinkageGenome.from_dict(gd)
    T = {k: np.asarray(v) * scale for k, v in compute_full_trajectory(g.nodes, g.edges, 360).items()}
    # links of Table 2: frame 0-1, crank 0-2 and the ten couplers (the 1-2 edge is not enforced by the solver)
    links = [(0, 1)] + [(e["n1"], e["n2"]) for e in gd["edges"] if {e["n1"], e["n2"]} != {1, 2}]
    foot, knee = T[gd["foot_id"]], T[gd["knee_id"]]
    y = foot[:, 1]
    low = y <= y.min() + band * np.ptp(y)
    allp = np.concatenate([T[k] for k in T])
    pad = 8
    xl, yl = (allp[:, 0].min() - pad, allp[:, 0].max() + pad), (allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1], hspace=0.12, wspace=0.08)
    for j, ang in enumerate(range(0, 360, 60)):
        a = fig.add_subplot(gs[0, j]); a.grid(False)
        a.plot(foot[:, 0], foot[:, 1], color="0.8", lw=0.8)
        for n1, n2 in links:
            p, q = T[n1][ang], T[n2][ang]
            ls = "--" if (n1, n2) == (0, 1) else "-"
            c = "#7f8c8d" if (n1, n2) == (0, 1) else ("#e67e22" if {n1, n2} == {0, 2} else "#2c3e50")
            a.plot([p[0], q[0]], [p[1], q[1]], ls, color=c, lw=1.6, solid_capstyle="round")
        for k, P in T.items():
            fixed = k in (0, 1)
            a.plot(*P[ang], "s" if fixed else "o", ms=4.5 if fixed else 3.2,
                   color="k" if fixed else "white", mec="k", mew=0.7)
        a.plot(*foot[ang], "o", ms=6, color=C_FOOT, mec="k", mew=0.6)
        a.plot(*knee[ang], "o", ms=5, color=C_KNEE, mec="k", mew=0.6)
        a.set_xlim(xl); a.set_ylim(yl); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True); s.set_color("0.75")
        a.set_title(f"{ang}°", fontsize=10)
    a = fig.add_subplot(gs[1, 1:5])
    a.plot(np.append(foot[:, 0], foot[0, 0]), np.append(foot[:, 1], foot[0, 1]), color="0.6", lw=0.8, zorder=1)
    a.scatter(foot[low, 0], foot[low, 1], s=9, color="#1e8449", zorder=2,
              label=f"Low (within {int(100*band)}% of the vertical range): {100*low.mean():.0f}% of the cycle")
    a.scatter(foot[~low, 0], foot[~low, 1], s=9, color="#e67e22", zorder=2,
              label=f"Raised: {100*(1-low.mean()):.0f}% of the cycle")
    for ang in range(0, 360, 60):
        a.plot(*foot[ang], "o", ms=7, color="white", mec="k", mew=0.9, zorder=3)
        a.annotate(f"{ang}°", foot[ang], xytext=(0, -13 if ang == 60 else 7), textcoords="offset points",
                   ha="center", fontsize=8)
    stride = np.ptp(foot[low, 0])
    a.set_title(f"Foot point H over one crank revolution\nhorizontal range while low {stride:.1f} mm, "
                f"vertical range {np.ptp(y):.1f} mm", fontsize=10)
    a.set_xlabel("Forward position (mm, prototype scale)"); a.set_ylabel("Vertical position (mm)")
    a.set_aspect("equal"); a.set_ylim(y.min() - 12, y.max() + 30)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(os.path.join(out, "fig_gait_sequence_3d.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  low {100*low.mean():.1f}%  stride {stride:.1f} mm  vertical {np.ptp(y):.1f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "paper_figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (fig_convergence, fig_genome, fig_pareto, fig_layers, fig_gait_sequence):
        f(args.out); print("done:", f.__name__)
    print("figures in", args.out)


if __name__ == "__main__":
    main()
