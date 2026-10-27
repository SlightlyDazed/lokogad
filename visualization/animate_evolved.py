#!/usr/bin/env python3
"""
animate_evolved.py — Animación 3D del mecanismo evolucionado.

Graph-coloring Z-layers: barras que comparten pivote nunca están en la misma capa.
Trazas de referencia humana superpuestas.

Uso: python3 animate_evolved.py [--thick]
  --thick : versión con piezas más gruesas para mejor visualización
"""
import os, sys, json, math, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import load_reference_trajectories, RESULT_DIR

import trimesh
from trimesh.viewer.windowed import SceneViewer


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH-COLORING PARA Z-LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_z_layers(edges):
    """Asigna capas Z a barras: barras con pivote compartido en capas distintas."""
    n = len(edges)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            nodes_i = {edges[i]['n1'], edges[i]['n2']}
            nodes_j = {edges[j]['n1'], edges[j]['n2']}
            if nodes_i & nodes_j:
                conflicts[i].add(j)
                conflicts[j].add(i)

    layers = [None] * n
    for i in range(n):
        used = {layers[j] for j in conflicts[i] if layers[j] is not None}
        for k in range(n + 1):
            candidate = (k + 1) // 2 * (1 if k % 2 == 0 else -1)
            if candidate not in used:
                layers[i] = candidate
                break
    return layers


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = [
    [255, 70, 70, 255], [70, 200, 70, 255], [70, 70, 220, 255],
    [220, 200, 50, 255], [200, 70, 200, 255], [70, 200, 200, 255],
    [140, 140, 210, 255], [220, 140, 50, 255], [50, 140, 220, 255],
    [230, 50, 130, 255], [50, 230, 130, 255], [180, 100, 60, 255],
]


def run_animation(thick=False):
    # ── Dimensiones ───────────────────────────────────────────
    if thick:
        BAR_WIDTH = 25.0   # mm
        BAR_THICK = 10.0
        PIN_R = 8.0
        LAYER_PITCH = 18.0
        title_suffix = " (Thick)"
    else:
        BAR_WIDTH = 5.0
        BAR_THICK = 2.0
        PIN_R = 1.8
        LAYER_PITCH = 4.0
        title_suffix = ""

    # ── Cargar genoma ─────────────────────────────────────────
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    with open(genome_path) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)
    ref = load_reference_trajectories()

    trajs = compute_full_trajectory(g.nodes, g.edges)
    if trajs is None:
        print("[ERROR] Mecanismo no ensambla")
        return
    foot_id = g.foot_id or 7
    knee_id = g.knee_id or 6
    foot_traj = trajs[foot_id]

    # Offset para alinear centroides (flip X)
    mech_flipped = np.column_stack([-foot_traj[:, 0], foot_traj[:, 1]])
    offset = ref['foot'].mean(axis=0) - mech_flipped.mean(axis=0)
    OX, OY = offset[0] / 1000, offset[1] / 1000

    # ── Z-layers ──────────────────────────────────────────────
    z_layers = compute_z_layers(g.edges)

    node_bars = {n['id']: [] for n in g.nodes}
    for i, e in enumerate(g.edges):
        node_bars[e['n1']].append(i)
        node_bars[e['n2']].append(i)

    # ── Construir escena ──────────────────────────────────────
    scene = trimesh.Scene()
    bar_meshes = {}
    bar_z = {}

    for i, e in enumerate(g.edges):
        L = e['length']
        hw = BAR_WIDTH / 2
        z_mm = z_layers[i] * LAYER_PITCH

        body = trimesh.creation.box(extents=[L, BAR_WIDTH, BAR_THICK])
        body.apply_translation([L / 2, 0, 0])
        cap1 = trimesh.creation.cylinder(radius=hw, height=BAR_THICK, sections=16)
        cap2 = trimesh.creation.cylinder(radius=hw, height=BAR_THICK, sections=16)
        cap2.apply_translation([L, 0, 0])
        bar = trimesh.util.concatenate([body, cap1, cap2])
        bar.visual.face_colors = COLORS[i % len(COLORS)]
        bar.apply_scale(0.001)
        name = f"bar_{i}"
        bar_meshes[name] = (e['n1'], e['n2'])
        bar_z[name] = z_mm / 1000
        scene.add_geometry(bar.copy(), node_name=name, geom_name=name)

    # Pines
    pin_z_center = {}
    for n in g.nodes:
        nid = n['id']
        connected = node_bars[nid]
        if connected:
            z_vals = [z_layers[bi] * LAYER_PITCH for bi in connected]
            zmin = min(z_vals) - BAR_THICK / 2
            zmax = max(z_vals) + BAR_THICK / 2
            height = max(zmax - zmin, BAR_THICK * 2)
        else:
            zmin, zmax, height = -BAR_THICK, BAR_THICK, BAR_THICK * 2

        center_z = (zmin + zmax) / 2
        pin_z_center[nid] = center_z / 1000

        pin = trimesh.creation.cylinder(
            radius=PIN_R * 0.001, height=height * 0.001, sections=16)

        if n['type'] == 'ground':
            pin.visual.face_colors = [255, 255, 255, 230]
        elif n['type'] == 'crank':
            pin.visual.face_colors = [255, 50, 50, 230]
        elif n['type'] == 'output_foot':
            pin.visual.face_colors = [50, 150, 255, 230]
        elif n['type'] == 'output_knee':
            pin.visual.face_colors = [50, 255, 150, 230]
        else:
            pin.visual.face_colors = [255, 220, 50, 230]

        scene.add_geometry(pin, node_name=f"pin_{nid}", geom_name=f"pin_{nid}")

    # Trazas
    fx = -foot_traj[:, 0] / 1000 + OX
    fy = foot_traj[:, 1] / 1000 + OY
    pts3d = np.column_stack([fx, fy, np.full(len(fx), -0.015)])
    fp = trimesh.load_path(pts3d)
    fp.colors = [[230, 50, 50, 200]] * len(fp.entities)
    scene.add_geometry(fp, node_name="foot_trace", geom_name="foot_trace")

    rf = ref['foot'] / 1000
    rf3d = np.column_stack([rf[:, 0], rf[:, 1], np.full(len(rf), -0.018)])
    rp = trimesh.load_path(rf3d)
    rp.colors = [[50, 150, 255, 200]] * len(rp.entities)
    scene.add_geometry(rp, node_name="ref_foot", geom_name="ref_foot")

    rk = ref['knee'] / 1000
    rk3d = np.column_stack([rk[:, 0], rk[:, 1], np.full(len(rk), -0.018)])
    rkp = trimesh.load_path(rk3d)
    rkp.colors = [[50, 255, 150, 200]] * len(rkp.entities)
    scene.add_geometry(rkp, node_name="ref_knee", geom_name="ref_knee")

    # Suelo
    ground_y = min(min(fy), rf[:, 1].min()) - 0.01
    ground = trimesh.creation.box(extents=[3.0, 0.003, 0.3])
    ground.visual.face_colors = [40, 180, 40, 40]
    Tg = np.eye(4)
    Tg[1, 3] = ground_y
    ground.apply_transform(Tg)
    scene.add_geometry(ground, node_name="gnd", geom_name="gnd")

    # ── Callback ──────────────────────────────────────────────
    _state = {'start': None, 'speed': 30.0}

    def animate(scene):
        if _state['start'] is None:
            _state['start'] = time.time()
        t = time.time() - _state['start']
        angle = (t * _state['speed']) % 360
        positions = solve_linkage(g.nodes, g.edges, angle)
        if positions is None:
            return

        for name, (n1, n2) in bar_meshes.items():
            if n1 in positions and n2 in positions:
                p1, p2 = positions[n1], positions[n2]
                sx, sy = -p1[0] / 1000 + OX, p1[1] / 1000 + OY
                ex, ey = -p2[0] / 1000 + OX, p2[1] / 1000 + OY
                a = math.atan2(ey - sy, ex - sx)
                T = np.eye(4)
                T[0, 0], T[0, 1] = math.cos(a), -math.sin(a)
                T[1, 0], T[1, 1] = math.sin(a), math.cos(a)
                T[0, 3], T[1, 3], T[2, 3] = sx, sy, bar_z[name]
                scene.graph.update(name, matrix=T)

        for n in g.nodes:
            nid = n['id']
            if nid in positions:
                T = np.eye(4)
                T[0, 3] = -positions[nid][0] / 1000 + OX
                T[1, 3] = positions[nid][1] / 1000 + OY
                T[2, 3] = pin_z_center[nid]
                scene.graph.update(f"pin_{nid}", matrix=T)

    metrics = data.get('metrics', {})
    mode = "THICK" if thick else "NORMAL"
    print(f"\n{'='*60}")
    print(f"  Mecanismo Evolucionado — Animación ({mode})")
    print(f"{'='*60}")
    print(f"  Nodos: {len(g.nodes)}  Barras: {len(g.edges)}")
    print(f"  Foot error: {metrics.get('foot_error', '?')}")
    print(f"  Pos error:  {metrics.get('pos_error', '?')}")
    print(f"  Stride:     {metrics.get('stride_mm', '?')}mm")
    print(f"  Z-layers:   {z_layers}")
    print(f"  Bar dims:   {BAR_WIDTH}×{BAR_THICK}mm  Pin R={PIN_R}mm")
    print(f"  ROJO=pie mech  AZUL=ref pie  VERDE=ref rodilla")
    print(f"  Controles: arrastrar=rotar, scroll=zoom, shift+drag=pan")
    print(f"{'='*60}\n")

    SceneViewer(scene, callback=animate, callback_period=1 / 30,
                resolution=(1400, 800),
                window_title=f"Mecanismo Evolucionado{title_suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thick", action="store_true",
                        help="Barras y pines más gruesos para mejor visualización")
    args = parser.parse_args()
    run_animation(thick=args.thick)
