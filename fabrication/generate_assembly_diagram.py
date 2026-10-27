#!/usr/bin/env python3
"""
generate_assembly_diagram.py
Genera un PNG con el diagrama completo de ensamblaje del mecanismo Jansen.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch, Circle, Wedge
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import RESULT_DIR

# ─── Paleta ──────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT  = "#4fc3f7"
GOLD    = "#ffd54f"
GREEN   = "#69f0ae"
RED     = "#ff5252"
GRAY    = "#b0bec5"
WHITE   = "#eceff1"
LAYER_COLORS = {
    -3: "#7c4dff",
    -2: "#448aff",
    -1: "#00bcd4",
     0: "#69f0ae",
     1: "#ffd54f",
     2: "#ff6d00",
}

# ─── Datos del mecanismo (del último export) ──────────────────────────
TARGET_XY_MM  = 250.0
BAR_WIDTH     = 6.0
BAR_THICK     = 3.0
HOLE_R        = 2.4
PIN_R         = 2.0
LAYER_PITCH   = 6.6
PRINT_TOL     = 0.4

# Barras con capas corregidas
BARS = [
    dict(idx=0,  n1=0, n2=2, L=13.1,  layer= 0),
    dict(idx=1,  n1=2, n2=3, L=84.2,  layer=-1),
    dict(idx=2,  n1=1, n2=3, L=72.5,  layer= 0),
    dict(idx=3,  n1=1, n2=4, L=67.4,  layer=-1),
    dict(idx=4,  n1=2, n2=4, L=84.5,  layer= 1),
    dict(idx=5,  n1=3, n2=5, L=93.7,  layer= 1),
    dict(idx=6,  n1=1, n2=5, L=41.4,  layer=-2),
    dict(idx=7,  n1=5, n2=6, L=64.7,  layer= 0),
    dict(idx=8,  n1=6, n2=7, L=92.0,  layer=-1),
    dict(idx=9,  n1=4, n2=7, L=114.3, layer=-2),
    dict(idx=10, n1=2, n2=6, L=109.8, layer= 2),
    dict(idx=11, n1=1, n2=2, L=14.7,  layer=-3),
]

PINS = [
    dict(nid=0, ntype="ground",       h=6.0,  cz= 0.0),
    dict(nid=1, ntype="ground",       h=22.8, cz=-9.9),
    dict(nid=2, ntype="crank",        h=36.0, cz=-3.3),
    dict(nid=3, ntype="free",         h=16.2, cz= 0.0),
    dict(nid=4, ntype="output_knee",  h=22.8, cz=-3.3),
    dict(nid=5, ntype="free",         h=22.8, cz=-3.3),
    dict(nid=6, ntype="free",         h=22.8, cz= 3.3),
    dict(nid=7, ntype="output_foot",  h=9.6,  cz=-9.9),
]

NODE_LABELS = {0:"N0\nsuelo", 1:"N1\nsuelo", 2:"N2\ncigüeñal",
               3:"N3\nlibre", 4:"N4\nrodilla", 5:"N5\nlibre",
               6:"N6\nlibre", 7:"N7\npie"}

# ─── Dibujar barra individual ─────────────────────────────────────────
def draw_bar_profile(ax, L, bar_w, hole_r, color, alpha=1.0,
                     label=None, show_dims=True):
    hw = bar_w / 2.0
    # Cuerpo (rectangulo + semicírculos)
    from matplotlib.patches import FancyBboxPatch
    # Rectángulo central
    rect = mpatches.FancyBboxPatch(
        (hole_r, -hw), L - 2*hole_r, bar_w,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    # Semicírculos llenos
    for cx in [0, L]:
        c = Circle((cx, 0), hw, color=color, alpha=alpha, zorder=3)
        ax.add_patch(c)
        # Hueco
        h = Circle((cx, 0), hole_r, color=BG, alpha=1.0, zorder=4)
        ax.add_patch(h)
        # Borde hueco
        h2 = Circle((cx, 0), hole_r, fill=False, edgecolor=GRAY,
                     linewidth=0.8, zorder=5)
        ax.add_patch(h2)
    if show_dims:
        # Flecha de longitud
        ax.annotate("", xy=(L, -hw-1.8), xytext=(0, -hw-1.8),
                    arrowprops=dict(arrowstyle="<->",
                    color=GOLD, lw=0.8))
        ax.text(L/2, -hw-2.8, f"{L:.1f} mm", ha="center",
                va="top", color=GOLD, fontsize=6.5)
        # Flecha ancho
        ax.annotate("", xy=(L+1.5, hw), xytext=(L+1.5, -hw),
                    arrowprops=dict(arrowstyle="<->",
                    color=ACCENT, lw=0.8))
        ax.text(L+3.0, 0, f"Ø{bar_w:.0f}", ha="left",
                va="center", color=ACCENT, fontsize=5.5)
        # Radio hueco
        ax.annotate("", xy=(hole_r, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>",
                    color=RED, lw=0.8))
        ax.text(hole_r/2, 0.6, f"Ø{hole_r*2:.1f}", ha="center",
                va="bottom", color=RED, fontsize=5)
    if label:
        ax.text(L/2, hw+1.5, label, ha="center", va="bottom",
                color=WHITE, fontsize=6.5, fontweight="bold")

# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(22, 28), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.55, wspace=0.35,
                           top=0.96, bottom=0.03,
                           left=0.04, right=0.97)

    # ── TÍTULO ────────────────────────────────────────────────────────
    fig.text(0.5, 0.977, "MECANISMO DE THEO JANSEN — GUÍA DE ENSAMBLAJE",
             ha="center", va="top", fontsize=18, fontweight="bold",
             color=WHITE, fontfamily="monospace")
    fig.text(0.5, 0.967,
             "Escala impresión: 166×250 mm  ·  "
             f"BAR_WIDTH={BAR_WIDTH}mm  ·  BAR_THICK={BAR_THICK}mm  ·  "
             f"HOLE_R={HOLE_R}mm  ·  PIN_R={PIN_R}mm  ·  holgura={PRINT_TOL}mm",
             ha="center", va="top", fontsize=9, color=GRAY)

    # ══════════════════════════════════════════════════════════════════
    # PANEL 1 — TOPOLOGÍA (grafo de conexiones)
    # ══════════════════════════════════════════════════════════════════
    ax_topo = fig.add_subplot(gs[0, :2])
    ax_topo.set_facecolor(PANEL)
    ax_topo.set_title("① TOPOLOGÍA DEL MECANISMO", color=ACCENT,
                      fontsize=11, fontweight="bold", pad=8)

    # Cargar posición de referencia (ángulo 0°)
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    with open(genome_path) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)
    trajs = compute_full_trajectory(g.nodes, g.edges)
    all_x = np.concatenate([t[:, 0] for t in trajs.values()])
    all_y = np.concatenate([t[:, 1] for t in trajs.values()])
    SCALE = TARGET_XY_MM / max(all_x.max()-all_x.min(),
                                all_y.max()-all_y.min())
    cx = (all_x.max()+all_x.min())/2
    cy = (all_y.max()+all_y.min())/2

    pos0 = solve_linkage(g.nodes, g.edges, 0.0)

    def world(pid):
        return ((pos0[pid][0]-cx)*SCALE,
                (pos0[pid][1]-cy)*SCALE)

    # Dibujar barras coloreadas por capa
    for b in BARS:
        p1 = world(b['n1'])
        p2 = world(b['n2'])
        col = LAYER_COLORS.get(b['layer'], GRAY)
        ax_topo.plot([p1[0], p2[0]], [p1[1], p2[1]],
                     color=col, linewidth=3.5, solid_capstyle='round',
                     zorder=2, alpha=0.85)
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax_topo.text(mx, my, f"B{b['idx']:02d}\n{b['L']:.0f}mm",
                     ha="center", va="center",
                     fontsize=6, color=BG, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15",
                               fc=col, alpha=0.9, lw=0))

    # Dibujar nodos
    for nid in range(8):
        px, py = world(nid)
        ntype = [n['type'] for n in g.nodes if n['id']==nid][0]
        nc = (RED if 'ground' in ntype else
              GOLD if 'crank' in ntype else
              GREEN if 'output' in ntype else WHITE)
        ax_topo.scatter(px, py, s=160, color=nc, zorder=5,
                        edgecolors=BG, linewidths=1.5)
        ax_topo.text(px+3, py+3, f"N{nid}", fontsize=8,
                     color=nc, fontweight="bold", zorder=6)

    # Leyenda capas
    handles = [mpatches.Patch(color=c, label=f"Capa {l:+d}  (z={l*LAYER_PITCH:+.1f}mm)")
               for l, c in sorted(LAYER_COLORS.items())]
    ax_topo.legend(handles=handles, loc="lower right",
                   facecolor=BG, edgecolor=GRAY,
                   labelcolor=WHITE, fontsize=7.5, framealpha=0.9)
    ax_topo.set_aspect("equal")
    ax_topo.axis("off")

    # ══════════════════════════════════════════════════════════════════
    # PANEL 2 — LEYENDA NODOS
    # ══════════════════════════════════════════════════════════════════
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.set_facecolor(PANEL)
    ax_leg.set_title("② NODOS", color=ACCENT,
                     fontsize=11, fontweight="bold", pad=8)
    ax_leg.axis("off")

    node_colors = {0:RED, 1:RED, 2:GOLD, 3:WHITE,
                   4:GREEN, 5:WHITE, 6:WHITE, 7:GREEN}
    node_descs  = {0:"Suelo fijo", 1:"Suelo fijo",
                   2:"Cigüeñal (motor)", 3:"Libre",
                   4:"Rodilla", 5:"Libre",
                   6:"Libre", 7:"Pie (efector)"}

    for i, (nid, nc) in enumerate(node_colors.items()):
        y = 0.92 - i*0.115
        ax_leg.scatter([0.05], [y], s=120, color=nc, transform=ax_leg.transAxes,
                       zorder=5, edgecolors=BG, linewidths=1.5)
        ax_leg.text(0.14, y, f"N{nid}  —  {node_descs[nid]}",
                    transform=ax_leg.transAxes,
                    va="center", fontsize=9, color=WHITE)
    # PINS info
    for p in PINS:
        pin_c = (RED if 'ground' in p['ntype'] else
                 GOLD if 'crank' in p['ntype'] else
                 GREEN if 'output' in p['ntype'] else WHITE)

    # ══════════════════════════════════════════════════════════════════
    # PANEL 3 — CATÁLOGO DE BARRAS (4 por fila)
    # ══════════════════════════════════════════════════════════════════
    ax_bars = fig.add_subplot(gs[1, :])
    ax_bars.set_facecolor(PANEL)
    ax_bars.set_title("③ CATÁLOGO DE BARRAS (12 piezas — imprimir planas, espesor 3mm)",
                      color=ACCENT, fontsize=11, fontweight="bold", pad=8)
    ax_bars.axis("off")

    cols = 4
    rows = 3
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows

    for b in BARS:
        col_i = b['idx'] % cols
        row_i = b['idx'] // cols
        x0 = col_i * cell_w
        y0 = 1.0 - (row_i+1) * cell_h

        ax_sub = ax_bars.inset_axes([x0+0.01, y0+0.04,
                                     cell_w-0.02, cell_h-0.05])
        ax_sub.set_facecolor(BG)
        ax_sub.axis("off")

        col = LAYER_COLORS.get(b['layer'], GRAY)
        L = b['L']
        draw_bar_profile(ax_sub, L, BAR_WIDTH, HOLE_R, color=col, alpha=0.88)

        margin = max(L * 0.15, 8)
        ax_sub.set_xlim(-margin, L + margin + 8)
        ax_sub.set_ylim(-BAR_WIDTH/2 - 5.5, BAR_WIDTH/2 + 4)
        ax_sub.set_aspect("equal")

        # Título
        z = b['layer'] * LAYER_PITCH
        ax_sub.set_title(
            f"BAR_{b['idx']:02d}  ·  N{b['n1']}→N{b['n2']}"
            f"\nL = {L:.1f} mm   |   Capa {b['layer']:+d}  (z={z:+.1f}mm)",
            color=WHITE, fontsize=6.5, pad=2,
            loc="left")

    # ══════════════════════════════════════════════════════════════════
    # PANEL 4 — CATÁLOGO DE PINES
    # ══════════════════════════════════════════════════════════════════
    ax_pins = fig.add_subplot(gs[2, :])
    ax_pins.set_facecolor(PANEL)
    ax_pins.set_title("④ CATÁLOGO DE PINES / ARTICULACIONES (8 piezas — imprimir verticales)",
                      color=ACCENT, fontsize=11, fontweight="bold", pad=8)
    ax_pins.axis("off")

    cell_w = 1.0 / 8
    for p in PINS:
        x0 = p['nid'] * cell_w
        ax_sub = ax_pins.inset_axes([x0+0.005, 0.05, cell_w-0.01, 0.88])
        ax_sub.set_facecolor(BG)
        ax_sub.axis("off")

        pin_c = (RED if 'ground' in p['ntype'] else
                 GOLD if 'crank' in p['ntype'] else
                 GREEN if 'output' in p['ntype'] else WHITE)

        H, R = p['h'], PIN_R
        # Cilindro (vista frontal = rectángulo + elipses)
        rect = mpatches.FancyBboxPatch(
            (-R, 0), 2*R, H,
            boxstyle="square,pad=0",
            facecolor=pin_c, alpha=0.75,
            edgecolor=GRAY, linewidth=0.6)
        ax_sub.add_patch(rect)
        # Tapa inferior
        ell_b = mpatches.Ellipse((0, 0), 2*R, R*0.55,
                                  facecolor=pin_c, alpha=0.9,
                                  edgecolor=GRAY, linewidth=0.6)
        ax_sub.add_patch(ell_b)
        # Tapa superior
        ell_t = mpatches.Ellipse((0, H), 2*R, R*0.55,
                                  facecolor="#ffffff", alpha=0.25,
                                  edgecolor=GRAY, linewidth=0.6)
        ax_sub.add_patch(ell_t)

        # Cotas
        ax_sub.annotate("", xy=(R+1.5, H), xytext=(R+1.5, 0),
                        arrowprops=dict(arrowstyle="<->",
                        color=GOLD, lw=0.8))
        ax_sub.text(R+3.0, H/2, f"h={H:.1f}mm", va="center",
                    fontsize=6, color=GOLD)
        ax_sub.annotate("", xy=(R, -2.5), xytext=(0, -2.5),
                        arrowprops=dict(arrowstyle="<->",
                        color=ACCENT, lw=0.8))
        ax_sub.text(R/2, -4.0, f"Ø{PIN_R*2:.1f}",
                    ha="center", fontsize=5.5, color=ACCENT)

        ax_sub.set_xlim(-R-5, R+12)
        ax_sub.set_ylim(-8, H+10)
        ax_sub.set_aspect("equal")
        ax_sub.set_title(
            f"PIN_{p['nid']:02d}\nN{p['nid']} · {p['ntype']}\nz={p['cz']:+.1f}mm",
            color=WHITE, fontsize=6.5, pad=2)

    # ══════════════════════════════════════════════════════════════════
    # PANEL 5 — SECCIÓN TRANSVERSAL (vista lateral capas Z)
    # ══════════════════════════════════════════════════════════════════
    ax_sec = fig.add_subplot(gs[3, :2])
    ax_sec.set_facecolor(PANEL)
    ax_sec.set_title("⑤ SECCIÓN LATERAL — Apilamiento de capas Z",
                     color=ACCENT, fontsize=11, fontweight="bold", pad=8)

    unique_layers = sorted(set(b['layer'] for b in BARS))
    for b in BARS:
        z = b['layer'] * LAYER_PITCH
        col = LAYER_COLORS.get(b['layer'], GRAY)
        rect = mpatches.FancyBboxPatch(
            (0, z - BAR_THICK/2), b['L'], BAR_THICK,
            boxstyle="round,pad=0.3",
            facecolor=col, alpha=0.7,
            edgecolor=GRAY, linewidth=0.4)
        ax_sec.add_patch(rect)
        ax_sec.text(b['L']/2, z,
                    f"B{b['idx']:02d}", ha="center", va="center",
                    fontsize=5.5, color=BG, fontweight="bold")

    # Líneas de referencia de capa
    for lay in unique_layers:
        z = lay * LAYER_PITCH
        ax_sec.axhline(z, color=LAYER_COLORS.get(lay, GRAY),
                       linewidth=0.4, linestyle=":", alpha=0.5)
        ax_sec.text(-5, z, f"z={z:+.1f}mm\n(capa {lay:+d})",
                    va="center", ha="right", fontsize=6,
                    color=LAYER_COLORS.get(lay, GRAY))

    ax_sec.set_facecolor(BG)
    ax_sec.set_xlabel("Longitud de barra (mm)", color=GRAY, fontsize=8)
    ax_sec.set_ylabel("Capa Z (mm)", color=GRAY, fontsize=8)
    ax_sec.tick_params(colors=GRAY, labelsize=7)
    ax_sec.spines[:].set_color(GRAY)
    ax_sec.spines[:].set_linewidth(0.4)
    ax_sec.set_ylim(min(unique_layers)*LAYER_PITCH - 6,
                    max(unique_layers)*LAYER_PITCH + 6)
    ax_sec.set_xlim(-20, 130)

    # ══════════════════════════════════════════════════════════════════
    # PANEL 6 — GUÍA DE ENSAMBLAJE
    # ══════════════════════════════════════════════════════════════════
    ax_guide = fig.add_subplot(gs[3, 2])
    ax_guide.set_facecolor(PANEL)
    ax_guide.set_title("⑥ GUÍA DE ENSAMBLAJE", color=ACCENT,
                       fontsize=11, fontweight="bold", pad=8)
    ax_guide.axis("off")

    steps = [
        ("1.", "Imprimir barras planas (Z=3mm)"),
        ("2.", "Imprimir pines verticales (eje↑)"),
        ("",   ""),
        ("3.", "Ensamblar desde capa más baja:"),
        ("    →", "Capa −3: BAR_11 (N1-N2, 14.7mm)"),
        ("    →", "Capa −2: BAR_06 + BAR_09"),
        ("    →", "Capa −1: BAR_01 + BAR_03 + BAR_08"),
        ("    →", "Capa  0: BAR_00 + BAR_02 + BAR_07"),
        ("    →", "Capa +1: BAR_04 + BAR_05"),
        ("    →", "Capa +2: BAR_10 (N2-N6, 109.8mm)"),
        ("",   ""),
        ("4.", "Insertar pines por articulación:"),
        ("    •", f"PIN_R={PIN_R}mm → HOLE_R={HOLE_R}mm"),
        ("    •", f"Holgura: {HOLE_R-PIN_R}mm → ajuste deslizante"),
        ("",   ""),
        ("5.", "N2 = cigüeñal → conectar al motor"),
        ("6.", "N0, N1 = puntos fijos al chasis"),
        ("7.", "N7 = pie efector (punto de contacto)"),
    ]

    for i, (num, text) in enumerate(steps):
        y = 0.97 - i * 0.055
        c = (GOLD if num.startswith(("1","2","3","4","5","6","7"))
             else ACCENT if "→" in num
             else GRAY)
        ax_guide.text(0.02, y, num, transform=ax_guide.transAxes,
                      va="top", fontsize=8.5, color=c, fontweight="bold")
        ax_guide.text(0.18, y, text, transform=ax_guide.transAxes,
                      va="top", fontsize=8, color=WHITE)

    # ── Guardar ───────────────────────────────────────────────────────
    out_path = os.path.join(RESULT_DIR, "stl_for_print",
                            "assembly_diagram.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"\n✓ Diagrama guardado en:\n  {out_path}\n")
    return out_path


if __name__ == "__main__":
    main()
