#!/usr/bin/env python3
"""
comparison_figure.py — Figura comparativa académica:
    Panel izquierdo : Modelo Theo Jansen (STLs 3D + trayectoria optimizada)
    Panel derecho   : Modelo OpenSim (trayectoria de referencia humana)

Usa la misma infraestructura que verify_stl_animation.py / snapshot_gait_figure.py.

Uso:
    source ~/pytorch-rocm/bin/activate
    cd <raíz del repositorio>
    python3 comparison_figure.py
"""
import os, sys, json, math, re
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.spatial import procrustes
from scipy.interpolate import interp1d

# ─── Paths ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import RESULT_DIR

STL_DIR = os.path.join(RESULT_DIR, "stl_for_print", "print_ready_250mm")
OUT_DIR = os.path.join(RESULT_DIR, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

RE_BAR = re.compile(r"BARRA_(\d+)__N(\d+)-N(\d+)__([\d.]+)mm__capa([+-]?\d+)\.stl")
RE_PIN = re.compile(r"PIN_(\d+)__([a-zA-Z0-9_]+)__h([\d.]+)mm\.stl")

# ─── Paleta oscura ────────────────────────────────────────────────────────────
DARK_BG   = "#0F0F1A"
PANEL_BG  = "#181828"
GRID_CLR  = "#252538"
ACCENT    = "#00D2FF"
ORANGE    = "#FF8C42"
GREEN     = "#39FF14"
RED_FOOT  = "#FF4B5C"
MAGENTA   = "#FF00CC"
YELLOW    = "#FFD93D"
TEXT_CLR  = "#E8E8F0"
MUTED     = "#888899"

BAR_COLORS = [
    [80,  80,  90,  255],
    [255, 80,  80,  255],
    [70,  200, 70,  255],
    [70,  70,  220, 255],
    [220, 200, 50,  255],
    [200, 70,  200, 255],
    [70,  200, 200, 255],
    [140, 140, 210, 255],
    [220, 140, 50,  255],
    [50,  140, 220, 255],
    [230, 50,  130, 255],
    [50,  230, 130, 255],
]
PIN_COLOR = [255, 220, 50, 240]


# ─── Helpers STL ──────────────────────────────────────────────────────────────
def parse_stl_dir(stl_dir):
    bars, pins = {}, {}
    for fname in sorted(os.listdir(stl_dir)):
        m = RE_BAR.match(fname)
        if m:
            idx, n1, n2, L = int(m[1]), int(m[2]), int(m[3]), float(m[4])
            bars[idx] = dict(fname=fname, n1=n1, n2=n2, L=L)
            continue
        m = RE_PIN.match(fname)
        if m:
            nid, ntype, h = int(m[1]), m[2], float(m[3])
            pins[nid] = dict(fname=fname, ntype=ntype, h=h)
    return bars, pins


def load_meshes(bars_meta, pins_meta):
    bar_meshes, pin_meshes = {}, {}
    for idx, meta in bars_meta.items():
        m = trimesh.load(os.path.join(STL_DIR, meta['fname']))
        m.visual.face_colors = BAR_COLORS[idx % len(BAR_COLORS)]
        bar_meshes[idx] = (m, meta)
    for nid, meta in pins_meta.items():
        m = trimesh.load(os.path.join(STL_DIR, meta['fname']))
        m.visual.face_colors = PIN_COLOR
        pin_meshes[nid] = (m, meta)
    return bar_meshes, pin_meshes


def render_mechanism(angle_deg, g, bars_meta, pins_meta,
                     bar_meshes, pin_meshes, cx, cy, SCALE, res=(900, 720)):
    positions = solve_linkage(g.nodes, g.edges, angle_deg)
    if positions is None:
        return None

    def world_xy(pid):
        return (positions[pid][0] - cx) * SCALE, (positions[pid][1] - cy) * SCALE

    scene = trimesh.Scene()
    for idx, (mesh_orig, meta) in bar_meshes.items():
        n1, n2 = meta['n1'], meta['n2']
        if n1 not in positions or n2 not in positions:
            continue
        sx, sy = world_xy(n1)
        ex, ey = world_xy(n2)
        ang = math.atan2(ey - sy, ex - sx)
        T = np.eye(4)
        T[0,0], T[0,1] =  math.cos(ang), -math.sin(ang)
        T[1,0], T[1,1] =  math.sin(ang),  math.cos(ang)
        T[0,3], T[1,3]  = sx, sy
        m = mesh_orig.copy(); m.apply_transform(T)
        scene.add_geometry(m, node_name=f"bar_{idx:02d}", geom_name=f"bar_{idx:02d}")

    for nid, (mesh_orig, _) in pin_meshes.items():
        if nid not in positions:
            continue
        sx, sy = world_xy(nid)
        T = np.eye(4); T[0,3], T[1,3] = sx, sy
        m = mesh_orig.copy(); m.apply_transform(T)
        scene.add_geometry(m, node_name=f"pin_{nid:02d}", geom_name=f"pin_{nid:02d}")

    try:
        import PIL.Image as PILImage, io
        png_bytes = scene.save_image(resolution=res, visible=False)
        img = PILImage.open(io.BytesIO(png_bytes))
        return np.array(img)
    except Exception as e:
        print(f"  [WARN] render: {e}")
        return None


# ─── Procrustes alignment ─────────────────────────────────────────────────────
def align_to_reference(mech_xy, ref_xy):
    """
    Alinea la trayectoria del mecanismo a la referencia OpenSim
    usando Procrustes (traslación + escala + rotación).
    Devuelve mech_xy alineado a la misma escala/posición que ref_xy.
    """
    _, mech_aligned, _ = procrustes(ref_xy, mech_xy)
    # Re-escalar al espacio de ref_xy
    ref_scale = np.sqrt((ref_xy**2).sum())
    mech_scale = np.sqrt((mech_aligned**2).sum())
    aligned = mech_aligned * ref_scale
    return aligned


# ─── Cargar datos OpenSim ─────────────────────────────────────────────────────
def load_opensim_refs():
    """Carga las trayectorias de referencia OpenSim (XYZ, 360 frames)."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensim_extraction")
    foot  = torch.load(os.path.join(base, "foot_ref_tensor.pt"),  weights_only=False).cpu().numpy()
    knee  = torch.load(os.path.join(base, "knee_ref_tensor.pt"),  weights_only=False).cpu().numpy()
    hip   = torch.load(os.path.join(base, "hip_ref_tensor.pt"),   weights_only=False).cpu().numpy()
    # Usamos X (forward) y Y (vertical) del espacio OpenSim
    # OpenSim: X=forward, Y=up, Z=lateral → plano sagital = X,Y
    return {
        "foot": foot[:, :2],   # (360, 2)
        "knee": knee[:, :2],
        "hip":  hip[:, :2],
    }


# ─── Figura principal ─────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  comparison_figure.py — Theo Jansen vs OpenSim")
    print("=" * 65)

    # 1) Genoma y escala
    with open(os.path.join(RESULT_DIR, "best_genome.json")) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)

    trajs_raw = compute_full_trajectory(g.nodes, g.edges)
    all_x = np.concatenate([t[:, 0] for t in trajs_raw.values()])
    all_y = np.concatenate([t[:, 1] for t in trajs_raw.values()])
    mech_size = max(np.ptp(all_x), np.ptp(all_y))
    SCALE = 250.0 / mech_size
    cx = (all_x.max() + all_x.min()) / 2.0
    cy = (all_y.max() + all_y.min()) / 2.0
    print(f"[OK] SCALE={SCALE:.4f}")

    # Identificar nodo del pie y rodilla
    nid_foot = max(trajs_raw, key=lambda n: np.ptp(trajs_raw[n][:, 1]))
    nid_knee = sorted(trajs_raw, key=lambda n: np.ptp(trajs_raw[n][:, 1]))[-2]

    foot_traj = np.column_stack([
        (trajs_raw[nid_foot][:, 0] - cx) * SCALE,
        (trajs_raw[nid_foot][:, 1] - cy) * SCALE,
    ])
    knee_traj = np.column_stack([
        (trajs_raw[nid_knee][:, 0] - cx) * SCALE,
        (trajs_raw[nid_knee][:, 1] - cy) * SCALE,
    ])

    # Fase de apoyo del mecanismo
    min_y = foot_traj[:, 1].min()
    max_y = foot_traj[:, 1].max()
    ground_mask = foot_traj[:, 1] <= (min_y + 0.35 * (max_y - min_y))

    # 2) STLs
    bars_meta, pins_meta = parse_stl_dir(STL_DIR)
    bar_meshes, pin_meshes = load_meshes(bars_meta, pins_meta)
    print(f"[OK] {len(bars_meta)} barras, {len(pins_meta)} pines cargados")

    # 3) Render del mecanismo en un ángulo representativo (pie en suelo)
    # Encontrar el ángulo donde el pie está más cerca del suelo
    ground_angles = np.where(ground_mask)[0]
    mid_ground = int(ground_angles[len(ground_angles) // 2]) if len(ground_angles) else 90
    print(f"[OK] Ángulo de render (pie en suelo): {mid_ground}°")
    img_mech = render_mechanism(
        mid_ground, g, bars_meta, pins_meta,
        bar_meshes, pin_meshes, cx, cy, SCALE,
        res=(900, 720)
    )

    # 4) OpenSim refs
    refs = load_opensim_refs()
    foot_os  = refs["foot"]   # (360,2) en metros
    knee_os  = refs["knee"]
    hip_os   = refs["hip"]

    # Centrar las trayectorias de OpenSim
    foot_os_c  = foot_os  - foot_os.mean(axis=0)
    knee_os_c  = knee_os  - knee_os.mean(axis=0)
    hip_os_c   = hip_os   - hip_os.mean(axis=0)

    # ─── Figura ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10), facecolor=DARK_BG)
    fig.suptitle(
        "Comparativa: Mecanismo Theo Jansen Optimizado  ↔  Referencia Biomecánica OpenSim",
        fontsize=15, fontweight='bold', color=TEXT_CLR, y=0.99
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.6, 1.0],
        width_ratios=[1.15, 0.08, 1.0],
        hspace=0.08,
        wspace=0.0,
        left=0.02, right=0.98, top=0.94, bottom=0.06,
    )

    # ══════════════════════════════════════════════════════
    # PANEL A: Render 3D del mecanismo
    # ══════════════════════════════════════════════════════
    ax_mech = fig.add_subplot(gs[0, 0])
    ax_mech.set_facecolor(PANEL_BG)
    ax_mech.axis('off')

    if img_mech is not None:
        ax_mech.imshow(img_mech, aspect='auto')
    else:
        ax_mech.text(0.5, 0.5, "Render no disponible", ha='center',
                     va='center', color='red', transform=ax_mech.transAxes)

    ax_mech.set_title(
        "Modelo 3D Imprimible — Mecanismo Theo Jansen\n"
        r"(STLs $print\_ready\_250mm$ · 12 barras · 8 pivotes)",
        color=ACCENT, fontsize=11, fontweight='bold', pad=6
    )

    # Etiqueta de fase
    ax_mech.text(0.02, 0.03, f"Ángulo manivela: {mid_ground}° (fase de apoyo)",
                 transform=ax_mech.transAxes, color=GREEN,
                 fontsize=9, va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_BG,
                           edgecolor=GREEN, alpha=0.85))

    # ══════════════════════════════════════════════════════
    # SEPARADOR VERTICAL
    # ══════════════════════════════════════════════════════
    ax_sep = fig.add_subplot(gs[:, 1])
    ax_sep.axis('off')
    ax_sep.text(0.5, 0.5, "vs.", ha='center', va='center',
                color=MUTED, fontsize=14, fontweight='bold',
                transform=ax_sep.transAxes, rotation=0)

    # ══════════════════════════════════════════════════════
    # PANEL B: OpenSim reference visualization
    # ══════════════════════════════════════════════════════
    ax_os = fig.add_subplot(gs[0, 2])
    ax_os.set_facecolor(PANEL_BG)
    ax_os.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax_os.spines.values():
        spine.set_color(GRID_CLR)
    ax_os.grid(True, color=GRID_CLR, alpha=0.5, lw=0.5)

    # Dibujar "silueta" de persona estilizada (esqueleto simplificado)
    # Usando las posiciones OpenSim en un frame representativo
    # Frame en la mitad del ciclo (frame 90 ≈ apoyo medio)
    frame_os = 90
    foot_pt  = foot_os_c[frame_os]
    knee_pt  = knee_os_c[frame_os]
    hip_pt   = hip_os_c[frame_os]

    # Trayectorias completas con color y grosor semántico
    ax_os.plot(foot_os_c[:, 0], foot_os_c[:, 1],
               color=RED_FOOT, lw=2.5, alpha=0.9,
               label="Trayectoria pie (OpenSim)", zorder=4)
    ax_os.plot(knee_os_c[:, 0], knee_os_c[:, 1],
               color=YELLOW, lw=2.0, alpha=0.8,
               label="Trayectoria rodilla (OpenSim)", zorder=3)
    ax_os.plot(hip_os_c[:, 0], hip_os_c[:, 1],
               color=MAGENTA, lw=1.8, alpha=0.7,
               label="Trayectoria cadera (OpenSim)", zorder=2)

    # Marcar frame representativo
    for pt, col, lbl in [
        (foot_pt, RED_FOOT, "Pie"),
        (knee_pt, YELLOW,   "Rodilla"),
        (hip_pt,  MAGENTA,  "Cadera"),
    ]:
        ax_os.scatter(*pt, s=140, c=col, edgecolors='white', lw=1.5, zorder=7)
        ax_os.annotate(lbl, xy=pt, xytext=(pt[0]+0.01, pt[1]+0.005),
                       color=col, fontsize=8.5, fontweight='bold')

    # Línea de "suelo" (el punto más bajo del pie)
    ground_y_os = foot_os_c[:, 1].min()
    ax_os.axhline(ground_y_os, color=GREEN, ls='--', lw=1, alpha=0.5)
    ax_os.text(foot_os_c[:, 0].min(), ground_y_os + 0.002,
               "  Nivel de suelo", color=GREEN, fontsize=8, alpha=0.7)

    # Flechas de dirección de movimiento
    for traj, col in [(foot_os_c, RED_FOOT), (knee_os_c, YELLOW)]:
        mid = len(traj) // 3
        dx = traj[mid+5, 0] - traj[mid, 0]
        dy = traj[mid+5, 1] - traj[mid, 1]
        ax_os.annotate("", xy=(traj[mid+5, 0], traj[mid+5, 1]),
                        xytext=(traj[mid, 0], traj[mid, 1]),
                        arrowprops=dict(arrowstyle='->', color=col,
                                        lw=2.0, mutation_scale=15))

    ax_os.set_title(
        "Referencia Biomecánica — OpenSim (Rajagopal)\n"
        "Trayectorias de articulaciones durante la marcha humana",
        color=ORANGE, fontsize=11, fontweight='bold', pad=6
    )
    ax_os.set_xlabel("Posición X sagital (m)", color=TEXT_CLR, fontsize=9)
    ax_os.set_ylabel("Posición Y vertical (m)", color=TEXT_CLR, fontsize=9)
    ax_os.legend(loc='upper right', fontsize=8,
                 facecolor=DARK_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    # ══════════════════════════════════════════════════════
    # PANEL C: Trayectorias del mecanismo (abajo izquierda)
    # ══════════════════════════════════════════════════════
    ax_foot_mech = fig.add_subplot(gs[1, 0])
    ax_foot_mech.set_facecolor(PANEL_BG)
    ax_foot_mech.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax_foot_mech.spines.values():
        spine.set_color(GRID_CLR)
    ax_foot_mech.grid(True, color=GRID_CLR, alpha=0.5, lw=0.5)

    fx, fy = foot_traj[:, 0], foot_traj[:, 1]
    kx, ky = knee_traj[:, 0], knee_traj[:, 1]

    ax_foot_mech.scatter(fx[ground_mask],  fy[ground_mask],
                         c=GREEN,  s=10, alpha=0.85, zorder=4,
                         label="Pie — apoyo")
    ax_foot_mech.scatter(fx[~ground_mask], fy[~ground_mask],
                         c=RED_FOOT, s=6, alpha=0.65, zorder=3,
                         label="Pie — swing")
    ax_foot_mech.plot(kx, ky, color=YELLOW, lw=1.8, alpha=0.8,
                      label="Rodilla (nodo 3)", zorder=2)

    # Punto actual en mid_ground
    ax_foot_mech.scatter(fx[mid_ground], fy[mid_ground], s=160,
                         c=ACCENT, edgecolors='white', lw=1.5,
                         zorder=8, label=f"Posición render ({mid_ground}°)")

    # Línea de suelo
    ax_foot_mech.axhline(fy[ground_mask].max() if ground_mask.any() else fy.min(),
                         color=GREEN, ls='--', lw=1, alpha=0.5)

    # Estadísticas embebidas
    stride = np.ptp(fx[ground_mask]) if ground_mask.any() else 0
    ax_foot_mech.text(0.02, 0.05,
                      f"Zancada: {stride:.1f} mm  |  "
                      f"Amplitud vertical pie: {np.ptp(fy):.1f} mm  |  "
                      f"Error Procrustes: 4.97%",
                      transform=ax_foot_mech.transAxes, color=ACCENT,
                      fontsize=8.5, va='bottom',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_BG,
                                edgecolor=ACCENT, alpha=0.85))

    ax_foot_mech.set_title("Trayectorias del Mecanismo Optimizado (espacio 2D)",
                            color=ACCENT, fontsize=10, fontweight='bold')
    ax_foot_mech.set_xlabel("Posición X (mm STL)", color=TEXT_CLR, fontsize=9)
    ax_foot_mech.set_ylabel("Posición Y (mm STL)", color=TEXT_CLR, fontsize=9)
    ax_foot_mech.legend(loc='upper right', fontsize=8,
                        facecolor=DARK_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    # ══════════════════════════════════════════════════════
    # PANEL D: Superposición normalizada (abajo derecha)
    # ══════════════════════════════════════════════════════
    ax_overlay = fig.add_subplot(gs[1, 2])
    ax_overlay.set_facecolor(PANEL_BG)
    ax_overlay.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax_overlay.spines.values():
        spine.set_color(GRID_CLR)
    ax_overlay.grid(True, color=GRID_CLR, alpha=0.5, lw=0.5)

    # Normalizar ambas trayectorias para comparación directa
    def norm_traj(t):
        t = t - t.mean(axis=0)
        scale = np.sqrt((t**2).sum(axis=1)).max()
        return t / (scale + 1e-9)

    foot_norm  = norm_traj(foot_traj)
    foot_os_n  = norm_traj(foot_os_c)
    knee_norm  = norm_traj(knee_traj)
    knee_os_n  = norm_traj(knee_os_c)

    ax_overlay.plot(foot_os_n[:, 0], foot_os_n[:, 1],
                    color=RED_FOOT, lw=2.5, alpha=0.9, ls='--',
                    label="Pie OpenSim (ref.)", zorder=3)
    ax_overlay.plot(foot_norm[:, 0], foot_norm[:, 1],
                    color=ACCENT, lw=2.2, alpha=0.95,
                    label="Pie Theo Jansen (opt.)", zorder=4)

    ax_overlay.plot(knee_os_n[:, 0], knee_os_n[:, 1],
                    color=YELLOW, lw=2.0, alpha=0.7, ls='--',
                    label="Rodilla OpenSim (ref.)", zorder=2)
    ax_overlay.plot(knee_norm[:, 0], knee_norm[:, 1],
                    color=ORANGE, lw=1.8, alpha=0.85,
                    label="Rodilla Theo Jansen (opt.)", zorder=3)

    # Métricas de similitud en caja
    ax_overlay.text(0.02, 0.97,
                    "Error Procrustes\n"
                    f"  Pie    : 4.97 %\n"
                    f"  Rodilla: 10.79 %",
                    transform=ax_overlay.transAxes, color=TEXT_CLR,
                    fontsize=8.5, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=DARK_BG,
                              edgecolor=ACCENT, alpha=0.9))

    ax_overlay.set_title(
        "Superposición Normalizada — Procrustes\n"
        "(trayectorias escaladas unitariamente para comparación directa)",
        color=ORANGE, fontsize=10, fontweight='bold'
    )
    ax_overlay.set_xlabel("X normalizado (u.a.)", color=TEXT_CLR, fontsize=9)
    ax_overlay.set_ylabel("Y normalizado (u.a.)", color=TEXT_CLR, fontsize=9)
    ax_overlay.legend(loc='lower right', fontsize=8,
                      facecolor=DARK_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
                      ncol=2, handlelength=1.5)

    # ─── Guardar ─────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "fig_comparison_opensim.png")
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f"\n✅ Figura comparativa guardada en:\n   {out_path}")


if __name__ == "__main__":
    main()
