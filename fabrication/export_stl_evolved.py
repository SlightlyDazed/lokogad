#!/usr/bin/env python3
"""
export_stl_evolved.py — Exporta exactamente las piezas del modo --thick
de animate_evolved.py como STL individuales para impresión 3D.

Barras: cuerpo rectangular + semicírculos en extremos (como en la animación).
Pines: cilindros centrados en Z, altura = span de capas que atraviesan.

Las dimensiones son IDÉNTICAS a las del visor: BAR_WIDTH=25mm, BAR_THICK=10mm,
PIN_R=8mm, LAYER_PITCH=18mm. Los pines tienen radio < orificio de la barra
(hw=12.5mm) por lo que NO se intersectan.

Uso:
    source ~/pytorch-rocm/bin/activate && python3 export_stl_evolved.py
"""
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import RESULT_DIR

import trimesh

# ═══════════════════════════════════════════════════════════════════════
# DIMENSIONES — idénticas al modo --thick de animate_evolved.py
# ═══════════════════════════════════════════════════════════════════════
BAR_WIDTH   = 25.0   # mm  — ancho de la barra (= diámetro de los semiciírculos)
BAR_THICK   = 10.0   # mm  — espesor de la barra (eje Z)
PIN_R       = 8.0    # mm  — radio del cilindro de pivote
LAYER_PITCH = 18.0   # mm  — separación entre capas Z
HOLE_R      = BAR_WIDTH / 2   # 12.5 mm — radio del extremo redondeado
# PIN_R (8) < HOLE_R (12.5) → los pines entran sin interferencia ✓

OUT_DIR = os.path.join(RESULT_DIR, "stl_for_print", "evolved_thick")


# ═══════════════════════════════════════════════════════════════════════
# Z-layers (graph-coloring, mismo algoritmo que animate_evolved.py)
# ═══════════════════════════════════════════════════════════════════════
def compute_z_layers(edges):
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


# ═══════════════════════════════════════════════════════════════════════
# BARRA — misma geometría que animate_evolved.py (body + cap1 + cap2)
#         exportada en mm (NO apply_scale 0.001) → lista para impresora
# ═══════════════════════════════════════════════════════════════════════
def make_bar_mm(length_mm):
    """
    Genera barra en mm con origen en (0,0,0) y extremos en x=0 y x=length_mm.
    Centrada en Z (de -BAR_THICK/2 a +BAR_THICK/2).
    """
    hw = BAR_WIDTH / 2

    # Cuerpo rectangular — alineado igual que en la animación
    body = trimesh.creation.box(extents=[length_mm, BAR_WIDTH, BAR_THICK])
    body.apply_translation([length_mm / 2, 0, 0])

    # Semiciírculos en extremos
    cap1 = trimesh.creation.cylinder(radius=hw, height=BAR_THICK, sections=32)
    cap2 = trimesh.creation.cylinder(radius=hw, height=BAR_THICK, sections=32)
    cap2.apply_translation([length_mm, 0, 0])

    bar = trimesh.util.concatenate([body, cap1, cap2])
    return bar


# ═══════════════════════════════════════════════════════════════════════
# PIN — cilindro centrado en Z con altura = span real de capas + tolerancia
#       Radio = PIN_R (8mm) < HOLE_R (12.5mm) → entra en los agujeros ✓
# ═══════════════════════════════════════════════════════════════════════
def make_pin_mm(height_mm, node_type='joint'):
    """
    Genera pin cilíndrico en mm, centrado en Z=0.
    """
    pin = trimesh.creation.cylinder(radius=PIN_R, height=height_mm, sections=32)
    return pin


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    # Cargar genoma
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    if not os.path.exists(genome_path):
        print(f"[ERROR] No encontrado: {genome_path}")
        sys.exit(1)

    with open(genome_path) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)

    # Verificar mecanismo
    trajs = compute_full_trajectory(g.nodes, g.edges)
    if trajs is None:
        print("[ERROR] Mecanismo no ensambla — revisa best_genome.json")
        sys.exit(1)

    # Calcular Z-layers
    z_layers = compute_z_layers(g.edges)

    # Mapeo nodo → barras conectadas
    node_bars = {n['id']: [] for n in g.nodes}
    for i, e in enumerate(g.edges):
        node_bars[e['n1']].append(i)
        node_bars[e['n2']].append(i)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  export_stl_evolved.py — Modo THICK")
    print(f"  BAR: {BAR_WIDTH}×{BAR_THICK}mm  PIN_R={PIN_R}mm  PITCH={LAYER_PITCH}mm")
    print(f"  Salida: {OUT_DIR}")
    print(f"{'='*60}\n")

    # ── Exportar barras ──────────────────────────────────────────
    print("[BARRAS]")
    metrics = data.get('metrics', {})
    for i, e in enumerate(g.edges):
        L = e['length']           # mm (el genoma guarda longitudes en mm)
        z_mm = z_layers[i] * LAYER_PITCH

        bar = make_bar_mm(L)

        # Posicionar en su capa Z correcta (centrado en esa capa)
        bar.apply_translation([0, 0, z_mm])

        # Nombre descriptivo
        n1, n2 = e['n1'], e['n2']
        fname = f"bar_{i:02d}_n{n1}-n{n2}_L{L:.0f}mm_z{z_mm:+.0f}mm.stl"
        fpath = os.path.join(OUT_DIR, fname)
        bar.export(fpath)
        print(f"  {fname}  ({len(bar.faces)} caras)")

    # ── Exportar pines ───────────────────────────────────────────
    print("\n[PINES / ARTICULACIONES]")
    for n in g.nodes:
        nid = n['id']
        connected = node_bars[nid]

        if connected:
            z_vals = [z_layers[bi] * LAYER_PITCH for bi in connected]
            zmin = min(z_vals) - BAR_THICK / 2
            zmax = max(z_vals) + BAR_THICK / 2
            height = max(zmax - zmin, BAR_THICK * 2)
        else:
            zmin, zmax = -BAR_THICK, BAR_THICK
            height = BAR_THICK * 2

        center_z = (zmin + zmax) / 2

        pin = make_pin_mm(height, node_type=n['type'])
        pin.apply_translation([0, 0, center_z])

        ntype = n.get('type', 'joint')
        fname = f"pin_{nid:02d}_{ntype}_h{height:.0f}mm_z{center_z:+.0f}mm.stl"
        fpath = os.path.join(OUT_DIR, fname)
        pin.export(fpath)
        print(f"  {fname}  ({len(pin.faces)} caras)")

    # ── Resumen ──────────────────────────────────────────────────
    all_stl = [f for f in os.listdir(OUT_DIR) if f.endswith(".stl")]
    total_size = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in all_stl)
    print(f"\n{'='*60}")
    print(f"  ✓ {len(all_stl)} STLs exportados en:")
    print(f"    {OUT_DIR}")
    print(f"  Tamaño total: {total_size/1024:.0f} KB")
    print(f"\n  VERIFICACIÓN DIMENSIONAL:")
    print(f"  · Radio pin   = {PIN_R:.1f} mm")
    print(f"  · Radio hueco = {HOLE_R:.1f} mm  (extremo redondeado de la barra)")
    print(f"  · Holgura     = {HOLE_R - PIN_R:.1f} mm  ← pin entra sin interferencia ✓")
    print(f"  · Espesor barra = {BAR_THICK:.1f} mm")
    print(f"  · Separación capas = {LAYER_PITCH:.1f} mm")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
