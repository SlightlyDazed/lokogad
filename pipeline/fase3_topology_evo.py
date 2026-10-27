#!/usr/bin/env python3
"""
fase3_topology_evo.py — Evolución topológica de linkages planos.

Algoritmo evolutivo custom que muta tanto los PARÁMETROS como la TOPOLOGÍA
del mecanismo para replicar trayectorias de marcha humana.

Uso: python3 fase3_topology_evo.py [--n-gen 500] [--pop-size 5000] [--workers 0]
"""
import os, sys, argparse, time, logging, json, random, math
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from general_solver import compute_full_trajectory, find_output_nodes
from linkage_genome import (
    LinkageGenome, create_jansen_genome, create_simple_4bar,
    mutate, crossover
)
from config_jansen import (
    _trajectory_error, _position_error,
    load_reference_trajectories, RESULT_DIR
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("topo_evo")

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
_REF_DATA = None

def _init_worker(ref_data):
    global _REF_DATA
    _REF_DATA = ref_data


def evaluate_genome(genome_dict):
    """Evalúa un genoma serializado. Para multiprocessing."""
    g = LinkageGenome.from_dict(genome_dict)

    # Computar trayectorias
    trajs = compute_full_trajectory(g.nodes, g.edges, n_angles=360)
    if trajs is None:
        return {'fitness': 100.0, 'valid': False,
                'foot_error': 10.0, 'knee_error': 10.0,
                'pos_error': 10.0, 'stride_mm': 0.0,
                'n_nodes': len(g.nodes), 'n_edges': len(g.edges)}

    # Auto-detectar pie y rodilla
    foot_id = g.foot_id
    knee_id = g.knee_id

    # Si no están asignados, detectar automáticamente
    if foot_id is None or foot_id not in trajs:
        foot_id, knee_id = find_output_nodes(g.nodes, trajs)
    if knee_id is None or knee_id not in trajs:
        _, knee_id = find_output_nodes(g.nodes, trajs)

    if foot_id is None or foot_id not in trajs:
        return {'fitness': 100.0, 'valid': False,
                'foot_error': 10.0, 'knee_error': 10.0,
                'pos_error': 10.0, 'stride_mm': 0.0,
                'n_nodes': len(g.nodes), 'n_edges': len(g.edges)}

    foot_traj = trajs[foot_id]  # (360, 2)
    knee_traj = trajs[knee_id] if knee_id and knee_id in trajs else None

    # Stride
    stride_mm = foot_traj[:, 0].max() - foot_traj[:, 0].min()

    # Errores
    foot_err = _trajectory_error(foot_traj, _REF_DATA['foot'])
    foot_pos = _position_error(foot_traj, _REF_DATA['foot'])

    if knee_traj is not None:
        knee_err = _trajectory_error(knee_traj, _REF_DATA['knee'])
        knee_pos = _position_error(knee_traj, _REF_DATA['knee'])
    else:
        knee_err = 5.0
        knee_pos = 5.0

    # Fitness combinado (menor = mejor)
    # Pesos: forma del pie es lo más importante
    pos_error = 0.6 * foot_pos + 0.4 * knee_pos
    fitness = (0.40 * foot_err +
               0.25 * knee_err +
               0.30 * pos_error +
               0.05 * max(0, abs(stride_mm - 1000) / 1000))  # penalizar stride lejos de 1m

    # Pequeña bonificación por simplicidad (menos barras = más fabricable)
    complexity_penalty = 0.001 * max(0, len(g.edges) - 12)
    fitness += complexity_penalty

    return {
        'fitness': float(fitness),
        'valid': True,
        'foot_error': float(foot_err),
        'knee_error': float(knee_err),
        'pos_error': float(pos_error),
        'stride_mm': float(stride_mm),
        'n_nodes': len(g.nodes),
        'n_edges': len(g.edges),
        'foot_id': foot_id,
        'knee_id': knee_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# POBLACIÓN INICIAL
# ═══════════════════════════════════════════════════════════════════════════════

def create_initial_population(pop_size, rng):
    """Crea la población inicial a partir de semillas Jansen + variantes."""
    pop = []

    # 1) Jansen a varias escalas (60%)
    n_jansen = int(pop_size * 0.6)
    for i in range(n_jansen):
        scale = rng.uniform(5.0, 30.0)
        g = create_jansen_genome(scale)
        # Aplicar 1-3 mutaciones para diversidad
        n_muts = rng.randint(0, 3)
        for _ in range(n_muts):
            g = mutate(g, rng)
        pop.append(g)

    # 2) 4-bar simples con mutaciones para explorar topologías nuevas (20%)
    n_4bar = int(pop_size * 0.2)
    for i in range(n_4bar):
        scale = rng.uniform(10.0, 25.0)
        g = create_simple_4bar(scale)
        # Muchas mutaciones para generar diversidad topológica
        n_muts = rng.randint(3, 8)
        for _ in range(n_muts):
            g = mutate(g, rng)
        pop.append(g)

    # 3) Jansen con mutaciones topológicas fuertes (20%)
    n_topo = pop_size - n_jansen - n_4bar
    for i in range(n_topo):
        scale = rng.uniform(8.0, 25.0)
        g = create_jansen_genome(scale)
        # Mutaciones estructurales fuertes
        n_muts = rng.randint(4, 10)
        for _ in range(n_muts):
            g = mutate(g, rng)
        pop.append(g)

    return pop


# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITMO EVOLUTIVO
# ═══════════════════════════════════════════════════════════════════════════════

def tournament_select(pop, k=5, rng=None):
    """Selección por torneo."""
    candidates = rng.sample(pop, min(k, len(pop)))
    return min(candidates, key=lambda g: g.fitness if g.fitness is not None else 1e9)


def run_evolution(n_gen=500, pop_size=5000, n_workers=None, seed=42):
    """Ejecuta la evolución topológica."""
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    log.info("=" * 60)
    log.info("  Evolución Topológica de Linkages")
    log.info("=" * 60)
    log.info(f"  Población:    {pop_size}")
    log.info(f"  Generaciones: {n_gen}")
    log.info(f"  Workers:      {n_workers}")

    # Cargar referencias
    ref_data = load_reference_trajectories()
    _init_worker(ref_data)  # también en proceso principal

    # Pool
    pool = Pool(n_workers, initializer=_init_worker, initargs=(ref_data,))

    # Población inicial
    log.info("  Creando población inicial...")
    pop = create_initial_population(pop_size, rng)
    log.info(f"  {len(pop)} genomas creados")

    # Parámetros
    elite_frac = 0.10
    mutation_rate = 0.80
    crossover_rate = 0.20
    n_elite = max(1, int(pop_size * elite_frac))

    best_ever = None
    best_ever_fitness = float('inf')
    history = []
    t0 = time.time()

    for gen in range(1, n_gen + 1):
        # ── Evaluar ───────────────────────────────────────────────
        dicts = [g.to_dict() for g in pop]
        results = pool.map(evaluate_genome, dicts)

        # Asignar fitness
        n_valid = 0
        for g, r in zip(pop, results):
            g.fitness = r['fitness']
            if r['valid']:
                n_valid += 1
                if r.get('foot_id') is not None:
                    g.foot_id = r['foot_id']
                if r.get('knee_id') is not None:
                    g.knee_id = r['knee_id']

        # Ordenar
        pop.sort(key=lambda g: g.fitness)
        best = pop[0]
        best_r = results[dicts.index(pop[0].to_dict())] if pop else None

        # Re-evaluar el best para obtener métricas
        best_result = evaluate_genome(best.to_dict())

        if best.fitness < best_ever_fitness:
            best_ever = best.copy()
            best_ever_fitness = best.fitness

        history.append({
            'gen': gen,
            'best_fitness': best.fitness,
            'n_valid': n_valid,
            'best_foot': best_result['foot_error'],
            'best_knee': best_result['knee_error'],
            'best_pos': best_result['pos_error'],
            'best_stride': best_result['stride_mm'],
            'best_n_nodes': best_result['n_nodes'],
            'best_n_edges': best_result['n_edges'],
        })

        # Log
        if gen % 5 == 0 or gen == 1:
            dt = time.time() - t0
            speed = gen * pop_size / dt
            mean_fit = np.mean([g.fitness for g in pop[:pop_size//2]])

            # Distribución de topologías
            topo_sizes = [len(g.nodes) for g in pop]
            avg_nodes = np.mean(topo_sizes)

            log.info(
                f"  Gen {gen:4d} | "
                f"valid={n_valid:4d}/{pop_size} | "
                f"best={best.fitness:.4f} mean={mean_fit:.4f} | "
                f"foot={best_result['foot_error']:.4f} "
                f"knee={best_result['knee_error']:.4f} "
                f"pos={best_result['pos_error']:.4f} | "
                f"stride={best_result['stride_mm']:.0f}mm "
                f"nodes={best_result['n_nodes']} edges={best_result['n_edges']} | "
                f"avg_nodes={avg_nodes:.1f} | "
                f"{speed:.0f} eval/s"
            )

        # ── Reproducción ──────────────────────────────────────────
        new_pop = []

        # Elitismo: top N pasan directo
        for i in range(n_elite):
            new_pop.append(pop[i].copy())

        # Generar el resto
        while len(new_pop) < pop_size:
            if rng.random() < crossover_rate:
                p1 = tournament_select(pop, k=5, rng=rng)
                p2 = tournament_select(pop, k=5, rng=rng)
                child = crossover(p1, p2, rng=rng)
            else:
                parent = tournament_select(pop, k=5, rng=rng)
                child = parent.copy()

            # Mutar
            if rng.random() < mutation_rate:
                n_muts = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
                for _ in range(n_muts):
                    child = mutate(child, rng)

            child.fitness = None
            new_pop.append(child)

        pop = new_pop

    pool.close()
    pool.join()

    dt = time.time() - t0
    log.info(f"\n[OK] Evolución completada en {dt:.0f}s ({dt/60:.1f}min)")

    # ── Guardar resultados ────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Re-evaluar best_ever
    _init_worker(ref_data)
    best_result = evaluate_genome(best_ever.to_dict())

    log.info(f"\n{'='*60}")
    log.info(f"  MEJOR MECANISMO (topología evolucionada):")
    log.info(f"    Nodos:       {best_result['n_nodes']}")
    log.info(f"    Barras:      {best_result['n_edges']}")
    log.info(f"    Fitness:     {best_result['fitness']:.4f}")
    log.info(f"    Foot error:  {best_result['foot_error']:.4f}")
    log.info(f"    Knee error:  {best_result['knee_error']:.4f}")
    log.info(f"    Pos error:   {best_result['pos_error']:.4f}")
    log.info(f"    Stride:      {best_result['stride_mm']:.0f}mm")
    log.info(f"{'='*60}")

    # Guardar genoma
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    genome_data = best_ever.to_dict()
    genome_data['metrics'] = best_result
    with open(genome_path, 'w') as f:
        json.dump(genome_data, f, indent=2)
    log.info(f"  Genoma: {genome_path}")

    # Guardar historial
    hist_path = os.path.join(RESULT_DIR, "evolution_history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    log.info(f"  Historial: {hist_path}")

    # Guardar top-10
    top10_path = os.path.join(RESULT_DIR, "top10_genomes.json")
    top10 = []
    for g in pop[:10]:
        gd = g.to_dict()
        gd['metrics'] = evaluate_genome(g.to_dict())
        top10.append(gd)
    with open(top10_path, 'w') as f:
        json.dump(top10, f, indent=2)
    log.info(f"  Top-10: {top10_path}")

    return best_ever, history


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Topology Evolution for Linkages")
    parser.add_argument("--n-gen", type=int, default=500)
    parser.add_argument("--pop-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers (0=auto)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    w = args.workers if args.workers > 0 else None
    run_evolution(n_gen=args.n_gen, pop_size=args.pop_size,
                  n_workers=w, seed=args.seed)


if __name__ == "__main__":
    main()
