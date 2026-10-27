#!/usr/bin/env python3
"""
verify_stl_animation.py — Verifica los STLs de print_ready_250mm
animando el mecanismo con las piezas impresas reales.

Carga los STL exportados (barras con huecos + pines) y los anima
exactamente igual que animate_evolved.py.  Si la geometría es
coherente, las piezas encajarán sin interpenetrarse.

Uso:
    source ~/pytorch-rocm/bin/activate && python3 verify_stl_animation.py
"""
import os, sys, json, math, time, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import RESULT_DIR

import trimesh
from trimesh.viewer.windowed import SceneViewer

STL_DIR = os.path.join(RESULT_DIR, "stl_for_print", "print_ready_250mm")

# ═══════════════════════════════════════════════════════════════════════
# Parsear metadatos desde nombre de archivo
#   BARRA_00__N0-N2__13mm__capa0.stl  →  idx=0 n1=0 n2=2
#   PIN_02__ciguenhal_motor__h36mm.stl  →  nid=2 type=ciguenhal_motor
# ═══════════════════════════════════════════════════════════════════════
RE_BAR = re.compile(
    r"BARRA_(\d+)__N(\d+)-N(\d+)__([\d.]+)mm__capa([+-]?\d+)\.stl"
)
RE_PIN = re.compile(
    r"PIN_(\d+)__([a-zA-Z0-9_]+)__h([\d.]+)mm\.stl"
)


def parse_stl_dir(stl_dir):
    bars = {}   # idx → {fname, n1, n2, L, z}
    pins = {}   # nid → {fname, ntype, h, z}

    for fname in sorted(os.listdir(stl_dir)):
        m = RE_BAR.match(fname)
        if m:
            idx, n1, n2, L = int(m[1]), int(m[2]), int(m[3]), float(m[4])
            bars[idx] = dict(fname=fname, n1=n1, n2=n2, L=L, z=0.0)
            continue
        m = RE_PIN.match(fname)
        if m:
            nid, ntype, h = int(m[1]), m[2], float(m[3])
            pins[nid] = dict(fname=fname, ntype=ntype, h=h, z=0.0)

    return bars, pins


def main():
    # ── Cargar genoma ────────────────────────────────────────────
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    with open(genome_path) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)

    # ── Calcular escala (igual que export_stl_print_ready.py) ───
    trajs = compute_full_trajectory(g.nodes, g.edges)
    if trajs is None:
        print("[ERROR] Mecanismo no ensambla"); sys.exit(1)

    all_x = np.concatenate([t[:, 0] for t in trajs.values()])
    all_y = np.concatenate([t[:, 1] for t in trajs.values()])
    mech_size = max(all_x.max()-all_x.min(), all_y.max()-all_y.min())
    SCALE = 250.0 / mech_size   # mm_stl / mm_genome

    # Centro del mecanismo (para centrarlo en pantalla)
    cx = (all_x.max() + all_x.min()) / 2.0
    cy = (all_y.max() + all_y.min()) / 2.0

    print(f"\n{'='*60}")
    print(f"  verify_stl_animation.py")
    print(f"  SCALE = {SCALE:.4f}  ({1/SCALE:.1f}x reducción)")
    print(f"  STL_DIR: {STL_DIR}")
    print(f"{'='*60}")

    # ── Parsear STLs ─────────────────────────────────────────────
    bars_meta, pins_meta = parse_stl_dir(STL_DIR)
    print(f"\n[OK] Encontrados: {len(bars_meta)} barras, {len(pins_meta)} pines")

    # ── Cargar STLs en la escena ─────────────────────────────────
    scene = trimesh.Scene()

    # Barras:
    #   El STL ya incluye el z de la capa → T[2,3] = 0 en la animación.
    #   El bar empieza en x=0 (pivote n1) y termina en x=L mm.
    #   El transform de animación rota en XY y traslada a la posición de n1.
    bar_info = {}   # name → {n1, n2, z}

    for idx, meta in bars_meta.items():
        fpath = os.path.join(STL_DIR, meta['fname'])
        mesh = trimesh.load(fpath)
        name = f"bar_{idx:02d}"
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        bar_info[name] = {'n1': meta['n1'], 'n2': meta['n2'], 'z': meta['z']}

    # Pines:
    #   El STL ya incluye su center_z → T[2,3] = 0 en la animación.
    pin_info = {}   # name → nid

    for nid, meta in pins_meta.items():
        fpath = os.path.join(STL_DIR, meta['fname'])
        mesh = trimesh.load(fpath)
        name = f"pin_{nid:02d}"
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        pin_info[name] = nid

    print(f"  Escena cargada: {len(scene.geometry)} geometrías")

    # ── Callback de animación ────────────────────────────────────
    _state = {'start': None, 'speed': 30.0}   # 30 °/s

    def animate(scene):
        if _state['start'] is None:
            _state['start'] = time.time()
        t = time.time() - _state['start']
        angle = (t * _state['speed']) % 360.0

        positions = solve_linkage(g.nodes, g.edges, angle)
        if positions is None:
            return

        # Posiciones escaladas y centradas (en mm, mismo sistema que los STLs)
        def world_xy(pid):
            x =  (positions[pid][0] - cx) * SCALE
            y =  (positions[pid][1] - cy) * SCALE
            return x, y

        # Actualizar barras
        for name, info in bar_info.items():
            n1, n2 = info['n1'], info['n2']
            if n1 not in positions or n2 not in positions:
                continue

            sx, sy = world_xy(n1)
            ex, ey = world_xy(n2)
            angle_bar = math.atan2(ey - sy, ex - sx)

            # Rotación 2D en Z + traslación XY.
            # Z = 0 porque el STL ya tiene su capa horneada.
            T = np.eye(4)
            T[0, 0] = math.cos(angle_bar)
            T[0, 1] = -math.sin(angle_bar)
            T[1, 0] = math.sin(angle_bar)
            T[1, 1] = math.cos(angle_bar)
            T[0, 3] = sx
            T[1, 3] = sy
            T[2, 3] = 0.0   # ← z de la capa ya está en el STL
            scene.graph.update(name, matrix=T)

        # Actualizar pines
        for name, nid in pin_info.items():
            if nid not in positions:
                continue
            sx, sy = world_xy(nid)
            T = np.eye(4)
            T[0, 3] = sx
            T[1, 3] = sy
            T[2, 3] = 0.0   # ← center_z ya está en el STL
            scene.graph.update(name, matrix=T)

    print(f"\n  Controles: arrastrar=rotar, scroll=zoom, shift+drag=pan")
    print(f"  Cierra la ventana para terminar\n")

    SceneViewer(
        scene,
        callback=animate,
        callback_period=1.0 / 30.0,
        resolution=(1400, 800),
        window_title="VERIFICACIÓN STLs — print_ready_250mm",
    )


if __name__ == "__main__":
    main()
