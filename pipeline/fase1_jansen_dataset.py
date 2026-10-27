#!/usr/bin/env python3
"""
fase1_jansen_dataset.py — Genera dataset de entrenamiento para el GNN.

Crea N variantes del mecanismo de Theo Jansen con MN aleatorios,
resuelve la cinemática, calcula errores de trayectoria, y guarda
como dataset PyG.

Uso: python3 fase1_jansen_dataset.py [--n-samples 10000]
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from config_jansen import (
    MN_ORIGINAL, MN_LOWER, MN_UPPER, N_MN,
    compute_trajectories, evaluate_mechanism,
    load_reference_trajectories, mechanism_to_pyg,
    DATA_DIR, SCALE_FACTOR
)


def generate_one_sample(args):
    """Genera un sample: MN aleatorios → cinemática → fitness."""
    idx, ref_data_shared = args
    rng = np.random.RandomState(idx)

    # Estrategia de muestreo: mezcla de exploración y refinamiento
    strategy = rng.choice(['uniform', 'scaled_jitter', 'scale_only'], p=[0.4, 0.4, 0.2])

    if strategy == 'uniform':
        # Exploración uniforme en todo el rango
        mn = MN_LOWER + rng.rand(N_MN) * (MN_UPPER - MN_LOWER)

    elif strategy == 'scaled_jitter':
        # Partimos del original escalado + ruido
        scale = rng.uniform(5, 25)  # escalas que cubren la humana (~15x)
        mn = MN_ORIGINAL * scale * (1 + rng.randn(N_MN) * 0.3)
        mn = np.clip(mn, MN_LOWER, MN_UPPER)

    else:  # scale_only
        # Solo escalar uniformemente (explorar qué escala es correcta)
        scale = rng.uniform(1, 30)
        mn = MN_ORIGINAL * scale
        mn = np.clip(mn, MN_LOWER, MN_UPPER)

    # Evaluar
    result = evaluate_mechanism(mn, ref_data_shared)

    targets = np.array([
        result['foot_error'],
        result['knee_error'],
        result['hip_error'],
        result['stride_mm'] / 1000.0,  # normalizar a metros
        result['ground_ratio'],
    ], dtype=np.float32)

    return mn, targets, result['valid']


def main():
    parser = argparse.ArgumentParser(description="Generar dataset Jansen")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(DATA_DIR, "jansen_dataset.pt")

    print("=" * 60)
    print("  Fase 1: Generación de Dataset para GNN Jansen")
    print("=" * 60)
    print(f"  Muestras: {args.n_samples}")
    print(f"  Output:   {args.output}")

    # Cargar referencias
    print("\n[INFO] Cargando trayectorias de referencia...")
    ref_data = load_reference_trajectories()
    foot_xr = ref_data['foot'][:,0].max() - ref_data['foot'][:,0].min()
    knee_xr = ref_data['knee'][:,0].max() - ref_data['knee'][:,0].min()
    print(f"  Pie:     {ref_data['foot'].shape} — rango X: {foot_xr:.0f}mm")
    print(f"  Rodilla: {ref_data['knee'].shape} — rango X: {knee_xr:.0f}mm")
    print(f"  Cadera:  oscilación Z = {ref_data['hip_z_range_mm']:.1f}mm")

    # Generar muestras
    print(f"\n[INFO] Generando {args.n_samples} variantes del mecanismo...")
    t0 = time.time()

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 2)
    task_args = [(i, ref_data) for i in range(args.n_samples)]

    if n_workers > 1:
        print(f"  Usando {n_workers} workers")
        with Pool(n_workers) as pool:
            results = list(pool.map(generate_one_sample, task_args))
    else:
        results = [generate_one_sample(a) for a in task_args]

    dt = time.time() - t0
    print(f"  Generación completada en {dt:.1f}s ({args.n_samples/dt:.0f} muestras/s)")

    # Convertir a PyG
    print("\n[INFO] Convirtiendo a grafos PyG...")
    dataset = []
    n_valid = 0
    for mn, targets, valid in results:
        if valid:
            n_valid += 1
        data = mechanism_to_pyg(mn, targets)
        data.mn = torch.tensor(mn, dtype=torch.float)
        data.is_valid = torch.tensor([1.0 if valid else 0.0])
        dataset.append(data)

    # Estadísticas
    all_targets = np.array([r[1] for r in results])
    valid_mask = np.array([r[2] for r in results])

    print(f"\n{'='*60}")
    print(f"  Estadísticas del Dataset")
    print(f"{'='*60}")
    print(f"  Total muestras:  {len(dataset)}")
    print(f"  Válidas:         {n_valid} ({100*n_valid/len(dataset):.1f}%)")
    print(f"  Inválidas:       {len(dataset)-n_valid}")

    if n_valid > 0:
        v_targets = all_targets[valid_mask]
        names = ['foot_err', 'knee_err', 'hip_err', 'stride_m', 'ground_ratio']
        for i, name in enumerate(names):
            vals = v_targets[:, i]
            print(f"  {name:<15s}: min={vals.min():.4f}  max={vals.max():.4f}  "
                  f"mean={vals.mean():.4f}  std={vals.std():.4f}")

    # Guardar
    torch.save(dataset, args.output)
    print(f"\n[OK] Dataset guardado: {args.output} ({os.path.getsize(args.output)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
