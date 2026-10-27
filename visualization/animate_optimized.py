#!/usr/bin/env python3
"""
animate_optimized.py — Animación interactiva del Theo Jansen OPTIMIZADO.

Carga los MN del mejor mecanismo (results/best_mechanism.json),
genera barras con CadQuery, las superpone con la trayectoria de referencia
humana y anima en tiempo real con trimesh SceneViewer.

Uso: source ~/pytorch-rocm/bin/activate && python3 animate_optimized.py
"""
import os, sys, math, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from config_jansen import (
    solve_mechanism, compute_trajectories,
    load_reference_trajectories, PIVOT_NAMES, BAR_DEFS, RESULT_DIR
)

# ═══════════ Config ═══════════
BAR_WIDTH = 30.0     # mm (escalado para mecanismo ~24x)
BAR_THICK = 10.0
PIN_R = 8.0
LAYER_PITCH = 15.0   # mm entre capas Z

Z_LAYER = {
    "chassis":  0,
    "crank":    1,
    "link_CD":  2,   "link_BD": -1,
    "link_CE": -2,   "link_BE":  3,
    "link_DF":  4,   "link_BF": -3,
    "link_FG":  5,   "link_EG": -4,
    "link_GH":  3,   "link_EH": -1,
}

PIVOTS = {
    "A": ["chassis", "crank"],
    "B": ["chassis", "link_BD", "link_BE", "link_BF"],
    "C": ["crank", "link_CD", "link_CE"],
    "D": ["link_CD", "link_BD", "link_DF"],
    "E": ["link_BE", "link_CE", "link_EG", "link_EH"],
    "F": ["link_DF", "link_BF", "link_FG"],
    "G": ["link_FG", "link_EG", "link_GH"],
    "H": ["link_GH", "link_EH"],
}

COLORS = {
    "chassis": [140,140,150,255], "crank": [255,70,70,255],
    "link_CD": [70,200,70,255],   "link_BD": [70,70,220,255],
    "link_BE": [220,200,50,255],  "link_CE": [200,70,200,255],
    "link_DF": [70,200,200,255],  "link_BF": [140,140,210,255],
    "link_FG": [220,140,50,255],  "link_EG": [50,140,220,255],
    "link_GH": [230,50,130,255],  "link_EH": [50,230,130,255],
}


def load_best_mn():
    """Carga los MN optimizados de best_mechanism.json."""
    path = os.path.join(RESULT_DIR, "best_mechanism.json")
    if not os.path.exists(path):
        print(f"[ERROR] No encontrado: {path}")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    mn = np.array(data['magic_numbers'])
    print(f"[OK] MN cargados: scale≈{mn.mean()/np.array(data['magic_numbers_original']).mean():.1f}x")
    print(f"     foot_err={data['foot_error']:.4f}  stride={data['stride_mm']:.0f}mm")
    return mn


def gen_bar_mesh(length):
    """Genera mesh de barra con trimesh (sin CadQuery para velocidad)."""
    import trimesh
    hw, ht = BAR_WIDTH / 2, BAR_THICK / 2

    # Cuerpo rectangular con extremos redondeados (simplificado como box + cylinders)
    body = trimesh.creation.box(extents=[length, BAR_WIDTH, BAR_THICK])
    body.apply_translation([length / 2, 0, 0])

    cap1 = trimesh.creation.cylinder(radius=hw, height=BAR_THICK, sections=16)
    cap1.apply_translation([0, 0, 0])
    cap2 = trimesh.creation.cylinder(radius=hw, height=BAR_THICK, sections=16)
    cap2.apply_translation([length, 0, 0])

    bar = trimesh.util.concatenate([body, cap1, cap2])

    # Agujeros para pines (restamos cilindros)
    hole1 = trimesh.creation.cylinder(radius=PIN_R, height=BAR_THICK * 2, sections=16)
    hole2 = trimesh.creation.cylinder(radius=PIN_R, height=BAR_THICK * 2, sections=16)
    hole2.apply_translation([length, 0, 0])

    try:
        bar = bar.difference(hole1)
        bar = bar.difference(hole2)
    except Exception:
        pass  # Boolean ops not always available

    return bar


def run_animation(mn):
    """Animación interactiva con trimesh."""
    import trimesh
    from trimesh.viewer.windowed import SceneViewer

    print("\n[INFO] Preparando animación del mecanismo optimizado...")

    # Verificar que el mecanismo funciona
    trajs = compute_trajectories(mn)
    if trajs is None:
        print("[ERROR] El mecanismo no ensambla para todos los ángulos!")
        return

    # Cargar referencias humanas
    ref_data = load_reference_trajectories()

    # Bar lengths
    bar_names = ["chassis", "crank", "link_CD", "link_BD", "link_BE", "link_CE",
                 "link_DF", "link_BF", "link_FG", "link_EG", "link_GH", "link_EH"]
    bar_lengths = {}
    for bdef in BAR_DEFS:
        name, _, _, mn_idx = bdef
        if mn_idx == -1:
            bar_lengths[name] = math.sqrt(mn[0]**2 + mn[1]**2)
        else:
            bar_lengths[name] = mn[mn_idx]

    # Generar meshes
    meshes = {}
    for name in bar_names:
        L = bar_lengths.get(name, 100)
        m = gen_bar_mesh(L)
        m.visual.face_colors = COLORS.get(name, [180,180,180,255])
        m.apply_scale(0.001)  # mm → m para visualización
        meshes[name] = m

    scene = trimesh.Scene()

    # === Trayectoria del pie (mecanismo) ===
    foot_pts = trajs['H']  # (360, 2)
    fx, fy = foot_pts[:, 0] / 1000, foot_pts[:, 1] / 1000
    foot_z = Z_LAYER.get('link_GH', 0) * LAYER_PITCH / 1000
    pts3d = np.column_stack([fx, fy, np.full(len(fx), foot_z)])
    if len(pts3d) > 2:
        foot_path = trimesh.load_path(pts3d)
        foot_path.colors = [[230, 50, 50, 200]] * len(foot_path.entities)
        scene.add_geometry(foot_path, node_name="foot_trace", geom_name="foot_trace")

    # === Trayectoria de referencia humana (pie) ===
    ref_foot = ref_data['foot'] / 1000  # mm → m
    ref3d = np.column_stack([ref_foot[:, 0], ref_foot[:, 1], np.full(len(ref_foot), foot_z - 0.02)])
    if len(ref3d) > 2:
        ref_path = trimesh.load_path(ref3d)
        ref_path.colors = [[50, 150, 255, 200]] * len(ref_path.entities)
        scene.add_geometry(ref_path, node_name="ref_foot", geom_name="ref_foot")

    # === Trayectoria de referencia (rodilla) ===
    ref_knee = ref_data['knee'] / 1000
    knee_z = Z_LAYER.get('link_FG', 0) * LAYER_PITCH / 1000
    knee3d = np.column_stack([ref_knee[:, 0], ref_knee[:, 1], np.full(len(ref_knee), knee_z - 0.02)])
    if len(knee3d) > 2:
        knee_path = trimesh.load_path(knee3d)
        knee_path.colors = [[50, 255, 150, 200]] * len(knee_path.entities)
        scene.add_geometry(knee_path, node_name="ref_knee", geom_name="ref_knee")

    # === Suelo ===
    ground_y = min(fy) - 0.01
    ground = trimesh.creation.box(extents=[3.0, 0.005, 0.3])
    ground.visual.face_colors = [40, 180, 40, 60]
    Tg = np.eye(4); Tg[1, 3] = ground_y
    ground.apply_transform(Tg)
    scene.add_geometry(ground, node_name="ground", geom_name="ground")

    # === Barras ===
    for name in meshes:
        scene.add_geometry(meshes[name].copy(), node_name=name, geom_name=name)

    # === Pines de pivote ===
    pin_radius = PIN_R * 0.001 * 1.2
    for pname, bars in PIVOTS.items():
        layers = [Z_LAYER[b] for b in bars if b in Z_LAYER]
        if not layers:
            continue
        zmin = min(layers) * LAYER_PITCH / 1000 - BAR_THICK / 2000
        zmax = max(layers) * LAYER_PITCH / 1000 + BAR_THICK / 2000
        height = zmax - zmin
        pin = trimesh.creation.cylinder(radius=pin_radius, height=height, sections=16)
        pin.visual.face_colors = [255, 220, 50, 230]
        Tp = np.eye(4); Tp[2, 3] = (zmin + zmax) / 2
        pin.apply_transform(Tp)
        gname = f"pin_{pname}"
        scene.add_geometry(pin, node_name=gname, geom_name=gname)

    # === Animación ===
    _state = {'start': None, 'speed': 30.0}

    def bar_transform(p1, p2, name):
        sx, sy = p1[0] / 1000, p1[1] / 1000
        ex, ey = p2[0] / 1000, p2[1] / 1000
        angle = math.atan2(ey - sy, ex - sx)
        z = Z_LAYER.get(name, 0) * LAYER_PITCH / 1000
        T = np.eye(4)
        T[0, 0], T[0, 1] = math.cos(angle), -math.sin(angle)
        T[1, 0], T[1, 1] = math.sin(angle),  math.cos(angle)
        T[0, 3], T[1, 3], T[2, 3] = sx, sy, z
        return T

    def pin_transform(pt, pname):
        bars = PIVOTS[pname]
        layers = [Z_LAYER[b] for b in bars if b in Z_LAYER]
        zmin = min(layers) * LAYER_PITCH / 1000 - BAR_THICK / 2000
        zmax = max(layers) * LAYER_PITCH / 1000 + BAR_THICK / 2000
        T = np.eye(4)
        T[0, 3] = pt[0] / 1000
        T[1, 3] = pt[1] / 1000
        T[2, 3] = (zmin + zmax) / 2
        return T

    def animate(scene):
        if _state['start'] is None:
            _state['start'] = time.time()
        t = time.time() - _state['start']
        angle = (t * _state['speed']) % 360
        pts = solve_mechanism(mn, angle)
        if pts.get('H') is None:
            return

        for bdef in BAR_DEFS:
            name, p1n, p2n, _ = bdef
            p1, p2 = pts.get(p1n), pts.get(p2n)
            if p1 and p2:
                scene.graph.update(name, matrix=bar_transform(p1, p2, name))

        for pname in PIVOTS:
            pt = pts.get(pname)
            if pt:
                scene.graph.update(f"pin_{pname}", matrix=pin_transform(pt, pname))

    print(f"\n[OK] Animación lista:")
    print(f"  {len(meshes)} barras + {len(PIVOTS)} pivotes")
    print(f"  Trayectoria pie (ROJO) vs referencia humana (AZUL)")
    print(f"  Trayectoria rodilla referencia (VERDE)")
    print(f"\n  Controles: arrastrar=rotar, scroll=zoom, shift+drag=pan")
    print(f"  Cierra la ventana para terminar")

    SceneViewer(scene, callback=animate, callback_period=1/30,
                resolution=(1400, 800),
                window_title="Theo Jansen OPTIMIZADO — vs Referencia Humana")


def main():
    mn = load_best_mn()
    run_animation(mn)


if __name__ == "__main__":
    main()
