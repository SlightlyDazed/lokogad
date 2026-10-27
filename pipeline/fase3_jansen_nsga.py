#!/usr/bin/env python3
"""
fase3_jansen_nsga.py — NSGA-II para optimizar Theo Jansen (EXACT KINEMATICS).

Usa cinemática exacta (pura matemática, ~4000 evals/s) en vez de GNN surrogate.
El GNN fue descartado porque el optimizador explotaba regiones donde el surrogate
predecía errores bajos para mecanismos que en realidad no ensamblan.

3 objetivos: minimizar error de pie, rodilla y desviación del ground ratio ideal.

Uso: python3 fase3_jansen_nsga.py [--n-gen 500] [--pop-size 10000]
"""
import os, sys, argparse, time, logging, json
import numpy as np
from multiprocessing import Pool, cpu_count

os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from config_jansen import (
    N_MN, MN_LOWER, MN_UPPER, N_OBJ, N_CONSTR, POP_SIZE, N_GEN,
    RESULT_DIR, MN_ORIGINAL, mn_to_unit, unit_to_mn,
    evaluate_mechanism, load_reference_trajectories
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("fase3_jansen")

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL ref_data para multiprocessing (shared memory)
# ═══════════════════════════════════════════════════════════════════════════════
_REF_DATA = None

def _init_worker(ref_data):
    global _REF_DATA
    _REF_DATA = ref_data

def _eval_one(x):
    """Evalúa un individuo con cinemática exacta. Para Pool.map."""
    mn = unit_to_mn(x)
    result = evaluate_mechanism(mn, _REF_DATA)
    if not result['valid']:
        F = np.array([10.0, 10.0, 1.0, 10.0])
        G = np.array([1.0, 1.0])
    else:
        F = np.array([
            result['foot_error'],
            result['knee_error'],
            abs(result['ground_ratio'] - 0.55),
            result['pos_error'],
        ])
        G = np.array([
            -result['stride_mm'],  # stride > 0
            -1.0,                   # valid
        ])
    return F, G


# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II — VECTORIZED + MULTIPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def run_optimization(n_gen=N_GEN, pop_size=POP_SIZE, n_workers=None):
    """Ejecuta NSGA-II con cinemática exacta parallelizada."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.termination import get_termination

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    log.info("=" * 60)
    log.info("  Fase 3: NSGA-II — Optimización Theo Jansen")
    log.info("  (Cinemática Exacta + Multiprocessing)")
    log.info("=" * 60)
    log.info(f"  Objetivos:      {N_OBJ}")
    log.info(f"  Variables:      {N_MN}")
    log.info(f"  Población:      {pop_size}")
    log.info(f"  Generaciones:   {n_gen}")
    log.info(f"  Workers:        {n_workers}")

    ref_data = load_reference_trajectories()

    # Pool persistente con ref_data preloaded
    pool = Pool(n_workers, initializer=_init_worker, initargs=(ref_data,))

    # ── Problema vectorizado ──────────────────────────────────────────────
    class JansenExactProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=N_MN,
                n_obj=N_OBJ,
                n_ieq_constr=N_CONSTR,
                xl=np.zeros(N_MN),
                xu=np.ones(N_MN),
            )

        def _evaluate(self, X, out, *args, **kwargs):
            results = pool.map(_eval_one, X)
            F = np.array([r[0] for r in results])
            G = np.array([r[1] for r in results])
            out["F"] = F
            out["G"] = G

    problem = JansenExactProblem()

    # ── Población inicial ─────────────────────────────────────────────────
    from scipy.stats import qmc

    seeds = []
    for scale in np.arange(1, 35, 0.5):
        mn_scaled = np.clip(MN_ORIGINAL * scale, MN_LOWER, MN_UPPER)
        seeds.append(mn_to_unit(mn_scaled))
    # Variantes con jitter
    rng = np.random.RandomState(42)
    for scale in [10, 12, 14, 15, 16, 18, 20]:
        for _ in range(50):
            mn_j = MN_ORIGINAL * scale * (1 + rng.randn(N_MN) * 0.15)
            mn_j = np.clip(mn_j, MN_LOWER, MN_UPPER)
            seeds.append(mn_to_unit(mn_j))

    n_seeds = min(len(seeds), pop_size)
    seeds = seeds[:n_seeds]

    n_remaining = pop_size - len(seeds)
    if n_remaining > 0:
        sampler = qmc.LatinHypercube(d=N_MN, seed=42)
        lhs = sampler.random(n=n_remaining)
        X_init = np.vstack([seeds, lhs])
    else:
        X_init = np.array(seeds[:pop_size])

    from pymoo.core.population import Population
    initial_pop = Population.new("X", X_init)

    log.info(f"  Semillas:       {len(seeds)} ({n_seeds} escaladas + jitter)")

    # ── Algoritmo ─────────────────────────────────────────────────────────
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=initial_pop,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=15),
        eliminate_duplicates=True,
    )

    # ── Callback ──────────────────────────────────────────────────────────
    class ProgressCallback:
        def __init__(self):
            self.history = []
            self.t0 = time.time()

        def __call__(self, algorithm):
            F = algorithm.pop.get("F")
            G = algorithm.pop.get("G")
            feasible = np.all(G <= 0, axis=1)
            valid = F[:, 0] < 9.0  # not penalty
            n_valid = np.sum(valid)
            n_feas = np.sum(feasible & valid)

            self.history.append({
                'gen': algorithm.n_gen,
                'n_valid': int(n_valid),
                'n_feasible': int(n_feas),
            })

            if algorithm.n_gen % 10 == 0:
                dt = time.time() - self.t0
                speed = algorithm.n_gen * pop_size / dt

                if n_feas > 0:
                    mask = feasible & valid
                    best_foot = F[mask, 0].min()
                    best_knee = F[mask, 1].min()
                    best_pos = F[mask, 3].min() if F.shape[1] > 3 else 0
                    mean_foot = F[mask, 0].mean()
                    log.info(
                        f"  Gen {algorithm.n_gen:4d} | "
                        f"válidos={n_valid:5d} factibles={n_feas:5d}/{pop_size} | "
                        f"best_foot={best_foot:.4f} best_knee={best_knee:.4f} "
                        f"best_pos={best_pos:.4f} mean_foot={mean_foot:.4f} | "
                        f"{speed:.0f} eval/s"
                    )
                else:
                    log.info(
                        f"  Gen {algorithm.n_gen:4d} | "
                        f"válidos={n_valid:5d} factibles={n_feas:5d}/{pop_size}"
                    )

    callback = ProgressCallback()

    # ── Ejecutar ──────────────────────────────────────────────────────────
    log.info(f"\n[INFO] Iniciando optimización ({n_workers} cores)...")
    t0 = time.time()

    res = minimize(
        problem, algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=42,
        callback=callback,
        verbose=False,
    )

    dt = time.time() - t0
    pool.close()
    pool.join()

    log.info(f"\n[OK] Optimización completada en {dt:.0f}s ({dt/60:.1f}min)")

    if res.F is None or len(res.F) == 0:
        log.error("No se encontraron soluciones factibles.")
        return None

    # Filtrar soluciones válidas del Pareto
    valid_mask = res.F[:, 0] < 9.0
    n_valid_pareto = np.sum(valid_mask)
    log.info(f"  Frente de Pareto: {len(res.F)} total, {n_valid_pareto} válidos")

    if n_valid_pareto == 0:
        log.error("No hay soluciones válidas en el Pareto.")
        # Guardar de todas formas para diagnóstico
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.save(os.path.join(RESULT_DIR, "pareto_F_all.npy"), res.F)
        np.save(os.path.join(RESULT_DIR, "pareto_X_all.npy"), res.X)
        return None

    # ── Top-N válidos ─────────────────────────────────────────────────────
    valid_F = res.F[valid_mask]
    valid_X = res.X[valid_mask]
    sort_idx = np.argsort(valid_F[:, 0])
    n_top = min(50, len(sort_idx))
    top_X = valid_X[sort_idx[:n_top]]

    top_MN, top_F_exact = [], []
    for i, x in enumerate(top_X):
        mn = unit_to_mn(x)
        result = evaluate_mechanism(mn, ref_data)
        top_MN.append(mn)
        top_F_exact.append([result['foot_error'], result['knee_error'],
                            abs(result['ground_ratio'] - 0.55)])
        if i < 10:
            log.info(
                f"  Top {i+1:2d}: foot={result['foot_error']:.4f} "
                f"knee={result['knee_error']:.4f} "
                f"pos={result.get('pos_error',0):.4f} "
                f"ground={result['ground_ratio']:.2f} "
                f"stride={result['stride_mm']:.0f}mm "
                f"scale={mn.mean()/MN_ORIGINAL.mean():.1f}x"
            )

    top_MN = np.array(top_MN)
    top_F_exact = np.array(top_F_exact)

    # ── Guardar ───────────────────────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    np.save(os.path.join(RESULT_DIR, "pareto_F.npy"), valid_F)
    np.save(os.path.join(RESULT_DIR, "pareto_X.npy"), valid_X)
    np.save(os.path.join(RESULT_DIR, "top_MN.npy"), top_MN)
    np.save(os.path.join(RESULT_DIR, "top_F_exact.npy"), top_F_exact)

    best_idx = np.argmin(top_F_exact[:, 0])
    best_mn = top_MN[best_idx]
    best_result = evaluate_mechanism(best_mn, ref_data)
    best_info = {
        'magic_numbers': best_mn.tolist(),
        'magic_numbers_original': MN_ORIGINAL.tolist(),
        'scale_vs_original': (best_mn / MN_ORIGINAL).tolist(),
        'foot_error': best_result['foot_error'],
        'knee_error': best_result['knee_error'],
        'stride_mm': best_result['stride_mm'],
        'ground_ratio': best_result['ground_ratio'],
    }
    with open(os.path.join(RESULT_DIR, "best_mechanism.json"), 'w') as f:
        json.dump(best_info, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"  MEJOR MECANISMO:")
    log.info(f"    Foot error:    {best_result['foot_error']:.4f}")
    log.info(f"    Knee error:    {best_result['knee_error']:.4f}")
    log.info(f"    Stride:        {best_result['stride_mm']:.0f}mm")
    log.info(f"    Ground ratio:  {best_result['ground_ratio']:.2f}")
    log.info(f"    Scale factor:  {(best_mn / MN_ORIGINAL).mean():.1f}x")
    log.info(f"{'='*60}")
    log.info(f"  Resultados en: {RESULT_DIR}")

    return res, callback.history


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NSGA-II Jansen (Exact Kinematics)")
    parser.add_argument("--n-gen", type=int, default=N_GEN)
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--workers", type=int, default=0,
                        help="Número de workers (0=auto)")
    args = parser.parse_args()

    w = args.workers if args.workers > 0 else None
    run_optimization(n_gen=args.n_gen, pop_size=args.pop_size, n_workers=w)


if __name__ == "__main__":
    main()
