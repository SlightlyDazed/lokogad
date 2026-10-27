#!/usr/bin/env python3
"""
comparison_visual.py — Figura comparativa académica:
  Panel IZQUIERDO : Mecanismo Theo Jansen (eslabonamiento 2D + trayectorias analíticas)
  Panel DERECHO   : Pierna OpenSim (segmentos corporales + trayectorias originales)

Corrección clave: en este dataset OpenSim, Z=vertical (altura), X=forward, Y=lateral.
El plano sagital correcto es X (col 0) vs Z (col 2).
Todos los joints se centran respecto al MISMO origen (media de la cadera).

Uso:
    source ~/pytorch-rocm/bin/activate
    cd <raíz del repositorio>
    python3 comparison_visual.py
"""
import os, sys, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import RESULT_DIR

OUT_DIR = os.path.join(RESULT_DIR, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Paleta ───────────────────────────────────────────────────────────────────
DARK_BG  = "#0D0D1A"
PANEL_BG = "#13131F"
GRID     = "#1E1E30"
C_FOOT   = "#FF4B5C"
C_KNEE   = "#FFD93D"
C_HIP    = "#CC44FF"
C_LINK   = "#00D2FF"
C_FIXED  = "#39FF14"
C_CRANK  = "#FF8C42"
TEXT     = "#E8E8F0"
MUTED    = "#556677"


# ══════════════════════════════════════════════════════════════════════════════
# CARGAR MECANISMO
# ══════════════════════════════════════════════════════════════════════════════
def load_mechanism():
    with open(os.path.join(RESULT_DIR, "best_genome.json")) as f:
        g = LinkageGenome.from_dict(json.load(f))

    trajs = compute_full_trajectory(g.nodes, g.edges)
    all_x = np.concatenate([t[:, 0] for t in trajs.values()])
    all_y = np.concatenate([t[:, 1] for t in trajs.values()])
    cx = (all_x.max() + all_x.min()) / 2.0
    cy = (all_y.max() + all_y.min()) / 2.0
    mech_size = max(np.ptp(all_x), np.ptp(all_y))

    nid_foot = 7
    nid_knee = 4

    ground_ids = {n['id'] for n in g.nodes if n['type'] == 'ground'}
    crank_ids  = {n['id'] for n in g.nodes if n['type'] == 'crank'}

    return g, trajs, cx, cy, mech_size, nid_foot, nid_knee, ground_ids, crank_ids


# ══════════════════════════════════════════════════════════════════════════════
# CARGAR OPENSIM
# ══════════════════════════════════════════════════════════════════════════════
def load_opensim():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensim_extraction")
    foot = torch.load(f"{base}/foot_ref_tensor.pt",  weights_only=False).cpu().numpy()
    knee = torch.load(f"{base}/knee_ref_tensor.pt",  weights_only=False).cpu().numpy()
    hip  = torch.load(f"{base}/hip_ref_tensor.pt",   weights_only=False).cpu().numpy()
    # Plano sagital correcto: X (col 0) = forward, Z (col 2) = vertical/up
    return foot, knee, hip  # (360, 3) cada una


# ══════════════════════════════════════════════════════════════════════════════
# DIBUJAR MECANISMO (izquierdo)
# ══════════════════════════════════════════════════════════════════════════════
def draw_mechanism_panel(ax, g, trajs, cx, cy, mech_size,
                         nid_foot, nid_knee, ground_ids, crank_ids):
    scale = 1.0 / mech_size

    def xy(pos, nid):
        return ((pos[nid][0] - cx) * scale,
                (pos[nid][1] - cy) * scale)

    node_type_map = {n['id']: n['type'] for n in g.nodes}

    # ── Trayectoria completa del pie (fondo tenue) ──
    foot_tr = np.column_stack([
        (trajs[nid_foot][:, 0] - cx) * scale,
        (trajs[nid_foot][:, 1] - cy) * scale,
    ])
    knee_tr = np.column_stack([
        (trajs[nid_knee][:, 0] - cx) * scale,
        (trajs[nid_knee][:, 1] - cy) * scale,
    ])

    # Cerrar el loop de la trayectoria
    foot_loop = np.vstack([foot_tr, foot_tr[:1]])
    knee_loop = np.vstack([knee_tr, knee_tr[:1]])

    ax.plot(foot_loop[:, 0], foot_loop[:, 1],
            color=C_FOOT, lw=2.5, alpha=0.9, label="Trayectoria Pie (nodo H)", zorder=4)
    ax.plot(knee_loop[:, 0], knee_loop[:, 1],
            color=C_KNEE, lw=2.0, alpha=0.85, label=f"Trayectoria Rodilla (nodo {nid_knee})", zorder=3)

    # Fase de apoyo
    min_fy = foot_tr[:, 1].min()
    max_fy = foot_tr[:, 1].max()
    ground_mask = foot_tr[:, 1] <= (min_fy + 0.32 * (max_fy - min_fy))
    ax.scatter(foot_tr[ground_mask, 0], foot_tr[ground_mask, 1],
               c=C_FIXED, s=14, alpha=0.6, zorder=5, label="Fase apoyo (suelo)")

    # Flechitas de dirección en la trayectoria del pie
    n = len(foot_tr)
    for i in range(0, n, n // 7):
        j = (i + 5) % n
        dx = foot_tr[j, 0] - foot_tr[i, 0]
        dy = foot_tr[j, 1] - foot_tr[i, 1]
        if abs(dx) + abs(dy) > 1e-5:
            ax.annotate("", xy=(foot_tr[j, 0], foot_tr[j, 1]),
                        xytext=(foot_tr[i, 0], foot_tr[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=C_FOOT,
                                        lw=1.2, mutation_scale=12))

    # ── Frame del mecanismo ──
    mid_g = int(np.where(ground_mask)[0][len(np.where(ground_mask)[0]) // 2]) \
            if ground_mask.any() else 90
    positions = solve_linkage(g.nodes, g.edges, mid_g)

    if positions is not None:
        for edge in g.edges:
            n1, n2 = edge['n1'], edge['n2']
            if n1 not in positions or n2 not in positions:
                continue
            x1, y1 = xy(positions, n1)
            x2, y2 = xy(positions, n2)
            is_crank = (n1 in crank_ids or n2 in crank_ids)
            col = C_CRANK if is_crank else C_LINK
            lw  = 3.5   if is_crank else 2.2
            ax.plot([x1, x2], [y1, y2], color=col, lw=lw, alpha=0.85,
                    solid_capstyle='round', zorder=6,
                    path_effects=[pe.Stroke(linewidth=lw+2, foreground='#00000070'),
                                  pe.Normal()])

        for nid in positions:
            xn, yn = xy(positions, nid)
            nt = node_type_map.get(nid, 'free')
            if nt == 'ground':
                m, c, s = 's', C_FIXED,  90
            elif nt == 'crank':
                m, c, s = 'o', C_CRANK,  80
            else:
                m, c, s = 'o', C_LINK,   55
            ax.scatter(xn, yn, s=s, c=c, marker=m,
                       edgecolors='white', linewidths=0.8, zorder=7)

    # Línea de suelo
    ground_y_mech = foot_tr[ground_mask, 1].max() if ground_mask.any() else foot_tr[:, 1].min()
    ax.axhline(ground_y_mech, color=C_FIXED, ls='--', lw=1.2, alpha=0.5)
    ax.text(foot_tr[:, 0].min() + 0.01, ground_y_mech + 0.015,
            "Nivel de suelo", color=C_FIXED, fontsize=8.5, alpha=0.7)

    # Anotación de métricas
    stride = np.ptp(foot_tr[ground_mask, 0]) if ground_mask.any() else 0
    ax.text(0.02, 0.98,
            "Error Procrustes\n"
            f"  Pie    : 4.97 %\n"
            f"  Rodilla: 10.79 %\n"
            f"Zancada: {stride:.3f} u.",
            transform=ax.transAxes, color=TEXT, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#050510',
                      edgecolor=C_LINK, alpha=0.92))

    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='lower right', fontsize=9,
              facecolor=DARK_BG, edgecolor=GRID, labelcolor=TEXT, framealpha=0.9)
    return mid_g


# ══════════════════════════════════════════════════════════════════════════════
# DIBUJAR PIERNA OPENSIM (derecho)
# ══════════════════════════════════════════════════════════════════════════════
def draw_opensim_panel(ax, foot_raw, knee_raw, hip_raw):
    """
    Plano sagital correcto: X (col 0) = forward, Z (col 2) = vertical.
    Centro común = media de la cadera (punto más estable).
    """
    hip_center = hip_raw.mean(axis=0)   # centro común para todos

    # Extraer plano sagital X-Z con el MISMO centro
    def sagittal(arr):
        c = arr - hip_center  # mismo origen
        return np.column_stack([c[:, 0], c[:, 2]])  # X_fwd, Z_vert

    foot = sagittal(foot_raw)  # (360, 2)
    knee = sagittal(knee_raw)
    hip  = sagittal(hip_raw)

    # Escalar: la longitud de la pierna en Z (cadera-suelo) ~= hip Z media
    leg_height = abs(hip_raw.mean(axis=0)[2])  # ≈ 1.0 m
    foot /= leg_height
    knee /= leg_height
    hip  /= leg_height

    print(f"  [OpenSim] hip  X=[{hip[:,0].min():.3f},{hip[:,0].max():.3f}] "
          f"Z=[{hip[:,1].min():.3f},{hip[:,1].max():.3f}]")
    print(f"  [OpenSim] knee X=[{knee[:,0].min():.3f},{knee[:,0].max():.3f}] "
          f"Z=[{knee[:,1].min():.3f},{knee[:,1].max():.3f}]")
    print(f"  [OpenSim] foot X=[{foot[:,0].min():.3f},{foot[:,0].max():.3f}] "
          f"Z=[{foot[:,1].min():.3f},{foot[:,1].max():.3f}]")

    # ── Trayectorias completas ──
    ax.plot(hip[:,0],  hip[:,1],  color=C_HIP,  lw=2.2, alpha=0.85,
            label="Cadera (hip\\_r)", zorder=3)
    ax.plot(knee[:,0], knee[:,1], color=C_KNEE, lw=2.5, alpha=0.9,
            label="Rodilla (knee\\_r)", zorder=4)
    ax.plot(foot[:,0], foot[:,1], color=C_FOOT, lw=2.8, alpha=0.95,
            label="Pie (foot\\_r)", zorder=5)

    # Flechitas de dirección
    for traj, col in [(foot, C_FOOT), (knee, C_KNEE), (hip, C_HIP)]:
        n = len(traj)
        for i in range(0, n, n // 6):
            j = (i + 6) % n
            dx = traj[j, 0] - traj[i, 0]
            dz = traj[j, 1] - traj[i, 1]
            if abs(dx) + abs(dz) > 5e-4:
                ax.annotate("", xy=(traj[j, 0], traj[j, 1]),
                            xytext=(traj[i, 0], traj[i, 1]),
                            arrowprops=dict(arrowstyle='->', color=col,
                                            lw=1.2, mutation_scale=11))

    frame_stance = int(np.argmin(foot[:, 1]))
    
    # ── Imagen de la pierna biomecánica (generada por Gemini) ──
    # Para darle un look más realista y evitar las "figuras de palitos"
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'opensim_leg_mesh.png')  # opcional
    import PIL.Image as PILImage
    try:
        img = np.array(PILImage.open(img_path).convert('RGBA'))
        # Ajustar la extensión de la imagen para que encaje con las coordenadas anatómicas
        # X: [-0.65, 0.45], Z: [-1.15, 0.25] son aprox los límites de la pierna
        ax.imshow(img, extent=[-0.75, 0.45, -1.15, 0.25], zorder=1, aspect='auto', alpha=0.9)
    except Exception as e:
        print(f"  [WARN] No se pudo cargar la imagen de la pierna: {e}")

    # Articulaciones en el frame representativo
    hp, kp, fp = hip[frame_stance], knee[frame_stance], foot[frame_stance]
    for pt, col, sz in [(hp, C_HIP, 130), (kp, C_KNEE, 110), (fp, C_FOOT, 95)]:
        ax.scatter(*pt, s=sz, c=col, edgecolors='white', linewidths=1.2, zorder=9)

    ax.annotate("Cadera",  xy=hp, xytext=(hp[0]-0.10, hp[1]+0.02),
                color=C_HIP,  fontsize=9.5, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
    ax.annotate("Rodilla", xy=kp, xytext=(kp[0]+0.03, kp[1]+0.03),
                color=C_KNEE, fontsize=9.5, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
    ax.annotate("Pie",     xy=fp, xytext=(fp[0]+0.03, fp[1]-0.04),
                color=C_FOOT, fontsize=9.5, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    # Línea de suelo
    ground_z = foot[:, 1].min()
    ax.axhline(ground_z, color=C_FIXED, ls='--', lw=1.2, alpha=0.6)
    ax.text(foot[:, 0].min() + 0.01, ground_z + 0.02,
            "Nivel de suelo", color=C_FIXED, fontsize=8.5, alpha=0.7)

    # Anotación del modelo
    ax.text(0.02, 0.98,
            "Modelo: Rajagopal 2016\n"
            "Joints: hip\\_r, knee\\_r, foot\\_r\n"
            "Plano: X (forward) vs Z (vertical)\n"
            "360 frames / ciclo de marcha",
            transform=ax.transAxes, color=TEXT, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#050510',
                      edgecolor=C_HIP, alpha=0.92))

    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='lower right', fontsize=9,
              facecolor=DARK_BG, edgecolor=GRID, labelcolor=TEXT, framealpha=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# ESTILO EJE
# ══════════════════════════════════════════════════════════════════════════════
def style_ax(ax, title, title_color, xlabel, ylabel):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.55, lw=0.5)
    ax.set_title(title, color=title_color, fontsize=12.5, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  comparison_visual.py — Theo Jansen vs OpenSim")
    print("=" * 65)

    g, trajs, cx, cy, mech_size, nid_foot, nid_knee, ground_ids, crank_ids = load_mechanism()
    foot_raw, knee_raw, hip_raw = load_opensim()

    print(f"[OK] Mecanismo: foot=nodo{nid_foot}, knee=nodo{nid_knee}")
    print(f"[OK] OpenSim: {foot_raw.shape} cada joint")

    # ── FIGURA ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 9.5), facecolor=DARK_BG)
    fig.suptitle(
        "Comparación Visual: Mecanismo Theo Jansen Optimizado (GAD)  ⟷  "
        "Referencia Biomecánica OpenSim (Rajagopal 2016)",
        fontsize=14, fontweight='bold', color=TEXT, y=0.99
    )

    gs = GridSpec(1, 3, figure=fig,
                  width_ratios=[1.0, 0.04, 1.0],
                  wspace=0.0,
                  left=0.03, right=0.97, top=0.93, bottom=0.09)

    # ─── Panel izquierdo — Mecanismo ─────────────────────────────────────────
    ax_left = fig.add_subplot(gs[0, 0])
    style_ax(ax_left,
             "Mecanismo Theo Jansen Optimizado\n"
             "Trayectorias analíticas: error Procrustes pie 4.97 %",
             C_LINK, "X normalizado", "Y normalizado")
    draw_mechanism_panel(ax_left, g, trajs, cx, cy, mech_size,
                         nid_foot, nid_knee, ground_ids, crank_ids)

    # ─── Separador ───────────────────────────────────────────────────────────
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_mid.axis('off')
    ax_mid.text(0.5, 0.5, "vs.", ha='center', va='center',
                color=MUTED, fontsize=17, fontweight='bold',
                transform=ax_mid.transAxes)

    # ─── Panel derecho — OpenSim ─────────────────────────────────────────────
    ax_right = fig.add_subplot(gs[0, 2])
    style_ax(ax_right,
             "Referencia Biomecánica — OpenSim (Rajagopal 2016)\n"
             "Trayectorias de marcha humana — plano sagital X vs Z",
             C_HIP, "X — dirección forward (norm.)", "Z — vertical / altura (norm.)")
    draw_opensim_panel(ax_right, foot_raw, knee_raw, hip_raw)

    # ─── Leyenda global ──────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=C_FOOT,  label="Pie / Punto H"),
        mpatches.Patch(color=C_KNEE,  label="Rodilla"),
        mpatches.Patch(color=C_HIP,   label="Cadera (solo OpenSim)"),
        mpatches.Patch(color=C_CRANK, label="Manivela (solo Jansen)"),
        mpatches.Patch(color=C_LINK,  label="Barras / Eslabones (Jansen)"),
        mpatches.Patch(color=C_FIXED, label="Puntos fijos / Suelo"),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6,
               fontsize=9, facecolor=DARK_BG, edgecolor=GRID,
               labelcolor=TEXT, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.001))

    # ─── Guardar ─────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "fig_comparison_visual.png")
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f"\n✅ Figura guardada en:\n   {out_path}")


if __name__ == "__main__":
    main()
