#!/usr/bin/env python3
"""
fase4_jansen_export.py — Exporta los mejores candidatos del Pareto.

Para cada candidato: genera STL + STEP + URDF + animación comparativa.

Uso: python3 fase4_jansen_export.py [--top-n 5]
"""
import os, sys, math, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from config_jansen import (
    RESULT_DIR, MESH_DIR, MN_ORIGINAL,
    compute_trajectories, evaluate_mechanism,
    load_reference_trajectories, PIVOT_NAMES
)


def generate_bar_stl_step(name, length, output_dir):
    """Genera STL + STEP de una barra con CadQuery."""
    import cadquery as cq
    BAR_W, BAR_T, PIN_R = 8.0, 3.0, 2.5
    hw, ht = BAR_W / 2, BAR_T / 2

    bar = (cq.Workplane("XY")
           .moveTo(0, -hw).lineTo(length, -hw)
           .threePointArc((length + hw, 0), (length, hw))
           .lineTo(0, hw)
           .threePointArc((-hw, 0), (0, -hw))
           .close().extrude(BAR_T).translate((0, 0, -ht)))
    bar = (bar.faces(">Z").workplane()
           .pushPoints([(0, 0), (length, 0)]).circle(PIN_R).cutThruAll())

    stl_path = os.path.join(output_dir, f"{name}.stl")
    step_path = os.path.join(output_dir, f"{name}.step")
    cq.exporters.export(bar, stl_path)
    cq.exporters.export(bar, step_path, cq.exporters.ExportTypes.STEP)
    return stl_path, step_path


def generate_comparison_plot(mn, ref_data, output_path):
    """Genera plot comparativo: trayectoria optimizada vs referencia."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    trajs = compute_trajectories(mn)
    if trajs is None:
        print("  [WARN] Mecanismo no ensambla")
        return

    foot_mech = trajs['H']
    knee_mech = trajs['G']
    ref_foot = ref_data['foot']
    ref_knee = ref_data['knee']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Trayectorias Optimizadas vs Referencia Humana',
                 color='white', fontsize=14, fontweight='bold')

    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, color='#444')
        ax.tick_params(colors='#888')
        for sp in ax.spines.values():
            sp.set_color('#333')

    # Panel 1: Pie
    ax = axes[0]
    ax.set_title('Trayectoria del PIE', color='#e0e0e0')
    ax.plot(ref_foot[:, 0], ref_foot[:, 1], '-', color='#3498db',
            linewidth=2, alpha=0.7, label='Referencia humana')
    ax.plot(foot_mech[:, 0], foot_mech[:, 1], '-', color='#e74c3c',
            linewidth=2, alpha=0.9, label='Jansen optimizado')
    ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc')
    ax.set_xlabel('X (mm)', color='#aaa')
    ax.set_ylabel('Y (mm)', color='#aaa')

    # Panel 2: Rodilla
    ax = axes[1]
    ax.set_title('Trayectoria de la RODILLA', color='#e0e0e0')
    ax.plot(ref_knee[:, 0], ref_knee[:, 1], '-', color='#2ecc71',
            linewidth=2, alpha=0.7, label='Referencia humana')
    ax.plot(knee_mech[:, 0], knee_mech[:, 1], '-', color='#e74c3c',
            linewidth=2, alpha=0.9, label='Jansen optimizado')
    ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc')
    ax.set_xlabel('X (mm)', color='#aaa')
    ax.set_ylabel('Y (mm)', color='#aaa')

    # Panel 3: Esqueleto completo
    ax = axes[2]
    ax.set_title('Poses del Mecanismo', color='#e0e0e0')
    bar_defs = [('A','B'),('A','C'),('C','D'),('B','D'),('B','E'),('C','E'),
                ('D','F'),('B','F'),('F','G'),('E','G'),('G','H'),('E','H')]
    colors_list = ['#888','#e74c3c','#2ecc71','#3498db','#f1c40f','#9b59b6',
                   '#1abc9c','#e67e22','#e74c3c','#3498db','#e74c3c','#2ecc71']
    for i in range(0, 360, 30):
        angle = i * 360 / 360
        pts = {n: trajs[n][i] for n in PIVOT_NAMES}
        alpha = 0.1 + 0.6 * (i / 360)
        for (p1, p2), c in zip(bar_defs, colors_list):
            ax.plot([pts[p1][0], pts[p2][0]], [pts[p1][1], pts[p2][1]],
                    '-', color=c, alpha=alpha, linewidth=1.5)
    # Foot trace
    ax.plot(foot_mech[:, 0], foot_mech[:, 1], '-', color='#e74c3c',
            linewidth=2, alpha=0.8, label='Pie')
    ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc')
    ax.set_xlabel('X (mm)', color='#aaa')
    ax.set_ylabel('Y (mm)', color='#aaa')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Plot: {output_path}")


def export_candidate(idx, mn, ref_data, output_dir):
    """Exporta un candidato completo: meshes + URDF + plot."""
    from config_jansen import BAR_DEFS, MN_ORIGINAL
    cand_dir = os.path.join(output_dir, f"candidate_{idx:03d}")
    mesh_dir = os.path.join(cand_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    result = evaluate_mechanism(mn, ref_data)
    print(f"\n  Candidato {idx}: foot_err={result['foot_error']:.4f} "
          f"knee_err={result['knee_error']:.4f} stride={result['stride_mm']:.0f}mm")

    # Generar STL/STEP para cada barra
    for name, p1, p2, mn_idx in BAR_DEFS:
        if mn_idx == -1:
            length = math.sqrt(mn[0]**2 + mn[1]**2)
        else:
            length = mn[mn_idx]
        generate_bar_stl_step(name, length, mesh_dir)
        print(f"    {name:.<20s} L={length:7.1f}mm")

    # Plot comparativo
    plot_path = os.path.join(cand_dir, "comparison.png")
    generate_comparison_plot(mn, ref_data, plot_path)

    # Guardar MN
    info = {
        'magic_numbers': mn.tolist(),
        'scale_vs_original': (mn / MN_ORIGINAL).tolist(),
        'foot_error': result['foot_error'],
        'knee_error': result['knee_error'],
        'stride_mm': result['stride_mm'],
        'ground_ratio': result['ground_ratio'],
    }
    with open(os.path.join(cand_dir, "info.json"), 'w') as f:
        json.dump(info, f, indent=2)

    return cand_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exportar candidatos Jansen")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("  Fase 4: Exportación de Candidatos Optimizados")
    print("=" * 60)

    # Cargar resultados
    mn_path = os.path.join(RESULT_DIR, "top_MN.npy")
    f_path = os.path.join(RESULT_DIR, "top_F_exact.npy")

    if not os.path.exists(mn_path):
        print("[ERROR] No se encontraron resultados. Ejecuta fase3 primero.")
        return

    top_MN = np.load(mn_path)
    top_F = np.load(f_path)

    # Ordenar por foot_error
    sort_idx = np.argsort(top_F[:, 0])
    n = min(args.top_n, len(top_MN))

    ref_data = load_reference_trajectories()
    output_dir = os.path.join(RESULT_DIR, "exported")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n):
        idx = sort_idx[i]
        export_candidate(i + 1, top_MN[idx], ref_data, output_dir)

    print(f"\n[OK] {n} candidatos exportados en {output_dir}")

    # Abrir el mejor plot
    import subprocess
    best_plot = os.path.join(output_dir, "candidate_001", "comparison.png")
    if os.path.exists(best_plot):
        subprocess.Popen(['xdg-open', best_plot])


if __name__ == "__main__":
    main()
