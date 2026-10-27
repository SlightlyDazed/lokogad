#!/usr/bin/env python3
"""
check_and_fix_collisions.py
Diagnostica cruces de barras en la misma capa Z a lo largo del ciclo completo,
y recalcula las capas añadiendo conflictos geométricos al graph-coloring.

Uso:
    source ~/pytorch-rocm/bin/activate && python3 check_and_fix_collisions.py
"""
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import RESULT_DIR

# ─── utilidad: intersección 2-D de segmentos ────────────────────────
def _cross2(ax, ay, bx, by):
    return ax * by - ay * bx

def segments_cross_2d(p1, p2, p3, p4, tol=1e-9):
    """¿Se intersectan los segmentos p1-p2 y p3-p4 en el interior?"""
    d1x, d1y = p2[0]-p1[0], p2[1]-p1[1]
    d2x, d2y = p4[0]-p3[0], p4[1]-p3[1]
    denom = _cross2(d1x, d1y, d2x, d2y)
    if abs(denom) < tol:
        return False                        # paralelos
    rx, ry = p3[0]-p1[0], p3[1]-p1[1]
    t = _cross2(rx, ry, d2x, d2y) / denom
    u = _cross2(rx, ry, d1x, d1y) / denom
    return (tol < t < 1-tol) and (tol < u < 1-tol)   # interior estricto

# ─── z-layers original (solo adyacencia) ────────────────────────────
def compute_z_layers_basic(edges):
    n = len(edges)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if {edges[i]['n1'], edges[i]['n2']} & {edges[j]['n1'], edges[j]['n2']}:
                conflicts[i].add(j); conflicts[j].add(i)
    layers = [None]*n
    for i in range(n):
        used = {layers[j] for j in conflicts[i] if layers[j] is not None}
        for k in range(n+1):
            c = (k+1)//2 * (1 if k%2==0 else -1)
            if c not in used:
                layers[i] = c; break
    return layers, conflicts

# ─── z-layers mejorado (adyacencia + cruces geométricos) ─────────────
def compute_z_layers_safe(edges, trajs):
    n = len(edges)
    N_FRAMES = len(next(iter(trajs.values())))   # 360

    # 1) conflictos por adyacencia
    _, conflicts = compute_z_layers_basic(edges)

    # 2) conflictos geométricos: revisar cada par NO adyacente
    geo_conflicts = []
    for i in range(n):
        for j in range(i+1, n):
            if j in conflicts[i]:
                continue    # ya conflictivos, ignorar

            n1i, n2i = edges[i]['n1'], edges[i]['n2']
            n1j, n2j = edges[j]['n1'], edges[j]['n2']

            tni1 = trajs.get(n1i); tni2 = trajs.get(n2i)
            tnj1 = trajs.get(n1j); tnj2 = trajs.get(n2j)
            if any(t is None for t in [tni1, tni2, tnj1, tnj2]):
                continue

            crossed = False
            for f in range(N_FRAMES):
                p1 = (tni1[f,0], tni1[f,1])
                p2 = (tni2[f,0], tni2[f,1])
                p3 = (tnj1[f,0], tnj1[f,1])
                p4 = (tnj2[f,0], tnj2[f,1])
                if segments_cross_2d(p1, p2, p3, p4):
                    crossed = True
                    break

            if crossed:
                conflicts[i].add(j); conflicts[j].add(i)
                geo_conflicts.append((i, j))

    # 3) graph-coloring con conflictos expandidos
    layers = [None]*n
    for i in range(n):
        used = {layers[j] for j in conflicts[i] if layers[j] is not None}
        for k in range(n+1):
            c = (k+1)//2 * (1 if k%2==0 else -1)
            if c not in used:
                layers[i] = c; break

    return layers, geo_conflicts


def main():
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    with open(genome_path) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)

    trajs = compute_full_trajectory(g.nodes, g.edges)
    if trajs is None:
        print("[ERROR] Mecanismo no ensambla"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  check_and_fix_collisions.py")
    print(f"  Analizando {len(g.edges)} barras × 360 ángulos ...")
    print(f"{'='*60}\n")

    layers_old, _ = compute_z_layers_basic(g.edges)
    layers_new, geo = compute_z_layers_safe(g.edges, trajs)

    print(f"[DIAGNÓSTICO]")
    if geo:
        print(f"  ⚠  {len(geo)} par(es) de barras se cruzan geométricamente:")
        for i, j in geo:
            ei, ej = g.edges[i], g.edges[j]
            print(f"     bar_{i:02d} (n{ei['n1']}-n{ei['n2']}) ↔ "
                  f"bar_{j:02d} (n{ej['n1']}-n{ej['n2']})")
            print(f"       capa anterior: {layers_old[i]}  vs  {layers_old[j]}  "
                  f"→ {'MISMO' if layers_old[i]==layers_old[j] else 'distinto'}")
    else:
        print(f"  ✓  Sin cruces geométricos detectados.")

    print(f"\n[CAPAS Z]")
    print(f"  {'Bar':<6} {'Nodos':<10} {'Antes':>6} {'Ahora':>6}  {'Cambió'}")
    print(f"  {'-'*44}")
    changed = 0
    for i, e in enumerate(g.edges):
        old, new = layers_old[i], layers_new[i]
        flag = "  ← FIX" if old != new else ""
        if old != new: changed += 1
        print(f"  bar_{i:02d}  n{e['n1']}-n{e['n2']:<7}  {old:>6}  {new:>6} {flag}")

    print(f"\n  Capas corregidas: {changed} de {len(g.edges)}")
    n_layers = len(set(layers_new))
    print(f"  Total capas usadas: {n_layers}")

    # Guardar resultado para que export_stl_print_ready.py lo use
    out = {
        "z_layers": layers_new,
        "geo_conflicts": [[i,j] for i,j in geo],
        "n_geo_conflicts": len(geo)
    }
    out_path = os.path.join(RESULT_DIR, "z_layers_fixed.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Capas corregidas guardadas en:")
    print(f"     {out_path}")
    print(f"\n  Ejecuta ahora:  python3 export_stl_print_ready.py  (usará las capas corregidas)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
