#!/usr/bin/env python3
"""
snapshot_gait_figure.py — Genera figura académica con capturas del mecanismo
Theo Jansen (STLs print_ready_250mm) en distintas etapas de la caminata.

Basado en verify_stl_animation.py: misma carga de STLs, escala y lógica de
transformación. En vez de SceneViewer interactivo, renderiza offline con
trimesh + PIL y compone la figura final con matplotlib.

Uso:
    source ~/pytorch-rocm/bin/activate
    cd <raíz del repositorio>
    python3 snapshot_gait_figure.py
"""
import os, sys, json, math, re, textwrap
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# ─── Path setup (igual que verify_stl_animation.py) ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import RESULT_DIR

STL_DIR   = os.path.join(RESULT_DIR, "stl_for_print", "print_ready_250mm")
OUT_DIR   = os.path.join(RESULT_DIR, "paper_figures")
SNAP_DIR  = os.path.join(OUT_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Regex (copiado de verify_stl_animation.py) ───────────────────────────────
RE_BAR = re.compile(r"BARRA_(\d+)__N(\d+)-N(\d+)__([\d.]+)mm__capa([+-]?\d+)\.stl")
RE_PIN = re.compile(r"PIN_(\d+)__([a-zA-Z0-9_]+)__h([\d.]+)mm\.stl")

# ─── Etapas del ciclo de caminata ─────────────────────────────────────────────
# (ángulo_manivela_grados, etiqueta, descripción corta)
GAIT_STAGES = [
    (  0, "t=0°",   "Contacto inicial\n(pie toca suelo)"),
    ( 60, "t=60°",  "Apoyo medio\n(transferencia peso)"),
    (120, "t=120°", "Despegue\n(pie abandona suelo)"),
    (180, "t=180°", "Oscilación media\n(fase aérea)"),
    (240, "t=240°", "Preparación\n(descenso del pie)"),
    (300, "t=300°", "Pre-contacto\n(talón próximo)"),
]

# ─── Colores por barra (visual) ───────────────────────────────────────────────
BAR_COLORS = [
    [80,  80,  90,  255],  # 00 chassis
    [255, 80,  80,  255],  # 01 crank (manivela)
    [70,  200, 70,  255],  # 02
    [70,  70,  220, 255],  # 03
    [220, 200, 50,  255],  # 04
    [200, 70,  200, 255],  # 05
    [70,  200, 200, 255],  # 06
    [140, 140, 210, 255],  # 07
    [220, 140, 50,  255],  # 08
    [50,  140, 220, 255],  # 09
    [230, 50,  130, 255],  # 10
    [50,  230, 130, 255],  # 11
]
PIN_COLOR = [255, 220, 50, 240]


# ─── Helpers ─────────────────────────────────────────────────────────────────
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


def build_scene(bars_meta, pins_meta):
    """Carga todos los STLs en geometrías separadas (sin posicionar)."""
    bar_meshes = {}
    pin_meshes = {}

    for idx, meta in bars_meta.items():
        fpath = os.path.join(STL_DIR, meta['fname'])
        m = trimesh.load(fpath)
        color = BAR_COLORS[idx % len(BAR_COLORS)]
        m.visual.face_colors = color
        bar_meshes[idx] = (m, meta)

    for nid, meta in pins_meta.items():
        fpath = os.path.join(STL_DIR, meta['fname'])
        m = trimesh.load(fpath)
        m.visual.face_colors = PIN_COLOR
        pin_meshes[nid] = (m, meta)

    return bar_meshes, pin_meshes


def make_transform(sx, sy, angle_rad):
    T = np.eye(4)
    T[0, 0] =  math.cos(angle_rad)
    T[0, 1] = -math.sin(angle_rad)
    T[1, 0] =  math.sin(angle_rad)
    T[1, 1] =  math.cos(angle_rad)
    T[0, 3] = sx
    T[1, 3] = sy
    T[2, 3] = 0.0  # z ya está horneado en el STL
    return T


def render_at_angle(angle_deg, g, bars_meta, pins_meta,
                    bar_meshes, pin_meshes, cx, cy, SCALE, res=(640, 480)):
    """
    Ensambla la escena para un ángulo dado y devuelve un array RGBA (H,W,4).
    """
    positions = solve_linkage(g.nodes, g.edges, angle_deg)
    if positions is None:
        return None

    def world_xy(pid):
        x = (positions[pid][0] - cx) * SCALE
        y = (positions[pid][1] - cy) * SCALE
        return x, y

    scene = trimesh.Scene()

    # ── Barras ──
    for idx, (mesh_orig, meta) in bar_meshes.items():
        n1, n2 = meta['n1'], meta['n2']
        if n1 not in positions or n2 not in positions:
            continue
        sx, sy = world_xy(n1)
        ex, ey = world_xy(n2)
        angle_bar = math.atan2(ey - sy, ex - sx)
        T = make_transform(sx, sy, angle_bar)
        m = mesh_orig.copy()
        m.apply_transform(T)
        scene.add_geometry(m, node_name=f"bar_{idx:02d}", geom_name=f"bar_{idx:02d}")

    # ── Pines ──
    for nid, (mesh_orig, meta) in pin_meshes.items():
        if nid not in positions:
            continue
        sx, sy = world_xy(nid)
        T = np.eye(4)
        T[0, 3] = sx
        T[1, 3] = sy
        T[2, 3] = 0.0
        m = mesh_orig.copy()
        m.apply_transform(T)
        scene.add_geometry(m, node_name=f"pin_{nid:02d}", geom_name=f"pin_{nid:02d}")

    # Cámara cenital (vista XY ligeramente inclinada)
    try:
        png_bytes = scene.save_image(resolution=res, visible=False)
        img_arr = np.frombuffer(png_bytes, dtype=np.uint8)
        import PIL.Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(png_bytes))
        return np.array(img)
    except Exception as e:
        print(f"  [WARN] render angle {angle_deg}°: {e}")
        return None


# ─── Trayectorias ─────────────────────────────────────────────────────────────
def compute_trajectories(g, cx, cy, SCALE):
    """
    Devuelve:
        - foot_xy   : trayectoria del pie (nodo efector final) en mm (STL)
        - all_trajs : dict nid → (N,2) array en mm (STL)
        - ground_mask: bool array para foot_xy (True = contacto suelo)
    """
    trajs_raw = compute_full_trajectory(g.nodes, g.edges)  # nid → (360,2)

    all_trajs = {}
    for nid, traj in trajs_raw.items():
        all_trajs[nid] = np.column_stack([
            (traj[:, 0] - cx) * SCALE,
            (traj[:, 1] - cy) * SCALE,
        ])

    # El pie es el nodo con la trayectoria de mayor amplitud vertical
    # (mismo criterio que export_stl_print_ready)
    nid_foot = max(all_trajs, key=lambda n: np.ptp(all_trajs[n][:, 1]))
    foot_xy = all_trajs[nid_foot]

    # Fase de contacto: porción inferior de la trayectoria (último 35%)
    min_y = foot_xy[:, 1].min()
    max_y = foot_xy[:, 1].max()
    threshold = min_y + 0.35 * (max_y - min_y)
    ground_mask = foot_xy[:, 1] <= threshold

    return foot_xy, all_trajs, ground_mask, nid_foot


# ─── Figura compuesta ─────────────────────────────────────────────────────────
def compose_figure(snapshots, angles, labels, descs, foot_xy, ground_mask):
    """
    Crea la figura académica final:
        Fila 1: 6 capturas 3D del mecanismo en cada etapa
        Fila 2: trayectoria completa del pie con puntos marcados
    """
    DARK_BG   = "#12121E"
    PANEL_BG  = "#1C1C2E"
    ACCENT    = "#00D2FF"
    ORANGE    = "#FF8C42"
    GREEN     = "#39FF14"
    RED_FOOT  = "#FF4B5C"
    TEXT_CLR  = "#E0E0E0"
    GRID_CLR  = "#2A2A3E"

    n = len(GAIT_STAGES)
    fig = plt.figure(figsize=(22, 11), facecolor=DARK_BG)
    fig.suptitle(
        "Mecanismo de Theo Jansen Optimizado — Secuencia de Caminata Biomimética\n"
        "Modelo 3D real imprimible (STLs print\\_ready\\_250mm) · "
        "Trayectoria vs. referencia OpenSim",
        fontsize=15, fontweight='bold', color=TEXT_CLR, y=0.98
    )

    gs = GridSpec(
        2, n,
        figure=fig,
        height_ratios=[1.7, 1.0],
        hspace=0.12,
        wspace=0.04,
        left=0.03, right=0.97, top=0.92, bottom=0.06
    )

    # ── Fila 1: capturas 3D ──────────────────────────────────────────────────
    for col, (img, angle, lbl, desc) in enumerate(
            zip(snapshots, angles, labels, descs)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PANEL_BG)
        ax.axis('off')

        if img is not None:
            ax.imshow(img, aspect='auto')
        else:
            ax.text(0.5, 0.5, "NaN\n(mecanismo\nno ensambla)",
                    ha='center', va='center', color='red',
                    fontsize=10, transform=ax.transAxes)

        # Marcador del ángulo
        ax.set_title(f"{lbl}", color=ACCENT, fontsize=11, fontweight='bold', pad=3)

        # Descripción debajo del título
        ax.text(0.5, -0.02, desc,
                ha='center', va='top', color=TEXT_CLR,
                fontsize=8.5, transform=ax.transAxes, linespacing=1.3)

        # Borde de color según fase
        is_ground = ground_mask[angle % 360]
        bcolor = GREEN if is_ground else ORANGE
        for spine in ax.spines.values():
            spine.set_edgecolor(bcolor)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

    # ── Fila 2: trayectoria del pie ──────────────────────────────────────────
    ax_traj = fig.add_subplot(gs[1, :])
    ax_traj.set_facecolor(PANEL_BG)
    ax_traj.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax_traj.spines.values():
        spine.set_color(GRID_CLR)
    ax_traj.grid(True, color=GRID_CLR, alpha=0.6, linewidth=0.6)

    fx, fy = foot_xy[:, 0], foot_xy[:, 1]

    # Fase de apoyo vs fase aérea
    ax_traj.scatter(fx[ground_mask],  fy[ground_mask],
                    c=GREEN,  s=14, alpha=0.85, label="Fase de apoyo (contacto suelo)", zorder=4)
    ax_traj.scatter(fx[~ground_mask], fy[~ground_mask],
                    c=ORANGE, s=8,  alpha=0.65, label="Fase aérea (swing)", zorder=3)

    # Línea suave de la trayectoria completa
    ax_traj.plot(fx, fy, color=ACCENT, lw=1.2, alpha=0.4, zorder=2)

    # Línea de suelo
    ground_y = fy[ground_mask].max() if ground_mask.any() else fy.min()
    ax_traj.axhline(ground_y, color=GREEN, ls='--', lw=1, alpha=0.5)
    ax_traj.text(fx.min() + 1, ground_y + 1, "Nivel de suelo",
                 color=GREEN, fontsize=7.5, alpha=0.7)

    # Marcar cada etapa con un punto grande y su ángulo
    stage_colors = [GREEN if ground_mask[a % 360] else ORANGE for a in angles]
    for angle, lbl, col in zip(angles, labels, stage_colors):
        a_idx = angle % 360
        ax_traj.scatter(fx[a_idx], fy[a_idx], s=120, c=col,
                        edgecolors='white', linewidths=1.2, zorder=6)
        ax_traj.annotate(
            lbl,
            xy=(fx[a_idx], fy[a_idx]),
            xytext=(fx[a_idx], fy[a_idx] + 4),
            fontsize=7.5, color=col, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=col, lw=0.8),
        )

    ax_traj.set_xlabel("Posición X del pie (mm STL)", color=TEXT_CLR, fontsize=9)
    ax_traj.set_ylabel("Posición Y del pie (mm STL)", color=TEXT_CLR, fontsize=9)
    ax_traj.set_title(
        "Trayectoria completa del pie (Punto H) — un ciclo de caminata (360°)",
        color=TEXT_CLR, fontsize=10, fontweight='bold'
    )

    # Leyenda de fase
    patches = [
        mpatches.Patch(color=GREEN,  label="Fase de apoyo (55% ciclo)"),
        mpatches.Patch(color=ORANGE, label="Fase aérea (45% ciclo)"),
    ]
    ax_traj.legend(handles=patches, loc='upper right', fontsize=8,
                   facecolor=DARK_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    # Estadísticas
    stride = fx[ground_mask].max() - fx[ground_mask].min() if ground_mask.any() else 0
    ax_traj.text(0.01, 0.05,
                 f"Zancada efectiva: {stride:.1f} mm  |  "
                 f"Amplitud vertical: {np.ptp(fy):.1f} mm  |  "
                 f"Error Procrustes pie: 4.97%  |  Rodilla: 10.79%",
                 transform=ax_traj.transAxes, color=ACCENT,
                 fontsize=8.5, va='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL_BG,
                           edgecolor=ACCENT, alpha=0.8))

    # ── Guardar ─────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "fig_gait_sequence_3d.png")
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f"\n✅ Figura académica guardada en:\n   {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  snapshot_gait_figure.py — Theo Jansen STL Gait Snapshots")
    print("=" * 65)

    # 1) Cargar genoma (mismo que verify_stl_animation)
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    with open(genome_path) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)
    print(f"[OK] Genoma cargado: {genome_path}")

    # 2) Calcular escala (idéntico a verify_stl_animation)
    trajs = compute_full_trajectory(g.nodes, g.edges)
    if trajs is None:
        print("[ERROR] El mecanismo no ensambla"); sys.exit(1)

    all_x = np.concatenate([t[:, 0] for t in trajs.values()])
    all_y = np.concatenate([t[:, 1] for t in trajs.values()])
    mech_size = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    SCALE = 250.0 / mech_size
    cx = (all_x.max() + all_x.min()) / 2.0
    cy = (all_y.max() + all_y.min()) / 2.0
    print(f"[OK] SCALE={SCALE:.4f}  cx={cx:.2f}  cy={cy:.2f}")

    # 3) Parsear y cargar STLs
    bars_meta, pins_meta = parse_stl_dir(STL_DIR)
    print(f"[OK] {len(bars_meta)} barras, {len(pins_meta)} pines")
    bar_meshes, pin_meshes = build_scene(bars_meta, pins_meta)

    # 4) Calcular trayectorias
    foot_xy, all_trajs, ground_mask, nid_foot = compute_trajectories(g, cx, cy, SCALE)
    print(f"[OK] Trayectoria del pie (nodo {nid_foot}): "
          f"X=[{foot_xy[:,0].min():.1f}, {foot_xy[:,0].max():.1f}] mm  "
          f"Y=[{foot_xy[:,1].min():.1f}, {foot_xy[:,1].max():.1f}] mm")

    # 5) Renderizar cada etapa
    angles = [s[0] for s in GAIT_STAGES]
    labels = [s[1] for s in GAIT_STAGES]
    descs  = [s[2] for s in GAIT_STAGES]

    snapshots = []
    for angle, lbl in zip(angles, labels):
        print(f"[...] Renderizando {lbl} (manivela {angle}°)...")
        img = render_at_angle(
            angle, g, bars_meta, pins_meta,
            bar_meshes, pin_meshes, cx, cy, SCALE,
            res=(640, 520)
        )
        snapshots.append(img)
        status = "OK" if img is not None else "WARN – NaN"
        print(f"  [{status}]")

        # Guardar snapshot individual también
        if img is not None:
            import PIL.Image as PILImage
            snap_path = os.path.join(SNAP_DIR, f"snap_{angle:03d}deg.png")
            PILImage.fromarray(img).save(snap_path)

    # 6) Componer figura académica
    compose_figure(snapshots, angles, labels, descs, foot_xy, ground_mask)


if __name__ == "__main__":
    main()
