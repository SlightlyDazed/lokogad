# =============================================================================
# 02_aproximacion_fourier.py
# Aproximación de trayectorias articulares con series de Fourier truncadas.
#
# ── IDEA CENTRAL (PDS) ───────────────────────────────────────────────────────
# Cualquier señal periódica x(t) puede expresarse como suma de senos y cosenos:
#
#   x(θ) = a₀  +  Σ_{k=1}^{H} [ aₖ·cos(kθ) + bₖ·sin(kθ) ]
#
# donde θ ∈ [0, 2π] es el ángulo del ciclo (aquí, de manivela).
# H = número de armónicos (cuantos más, mejor la aproximación pero más parámetros).
# a₀  = componente DC  (media de la señal)
# aₖ, bₖ = amplitudes del k-ésimo armónico (parte par e impar)
#
# Aquí las trayectorias son 3D y para 3 joints → optimizamos
#   3 joints × 3 ejes × (1 + 2·H)  coeficientes en total.
#
# Estrategia de optimización:
#   1. Differential Evolution (DE) – búsqueda global sin gradiente
#   2. Adam – refinamiento local con gradiente (PyTorch autograd)
#
# Entrada : joint_trajectories_opensim.mat
# Salida  : fourier_results.pt  (lista de dicts, uno por configuración probada)
# =============================================================================

import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

# ── Dispositivo ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# =============================================================================
# SECCIÓN 1 — PREPARACIÓN DE LOS DATOS DE REFERENCIA
# =============================================================================

def cargar_y_normalizar(mat_path="joint_trajectories_opensim.mat", n_pts=360):
    """
    Carga el .mat de OpenSim, interpola a n_pts puntos y retorna tensores
    de referencia normalizados en [0,1] por eje.

    Retorna
    -------
    refs_norm : dict  { 'foot'|'knee'|'hip' → tensor (n_pts, 3) normalizado }
    refs_raw  : dict  { mismo, sin normalizar, en metros }
    stats     : dict  { joint → (mins, scales) para desnormalizar }
    """
    mat = sio.loadmat(mat_path)
    jt  = mat["joint_trajectories"]

    def extraer_interpolar(campo):
        arr = jt[campo][0, 0].astype(np.float32)   # (73, 3)
        t_in  = np.linspace(0, 1, arr.shape[0])
        t_out = np.linspace(0, 1, n_pts)
        f = interp1d(t_in, arr, axis=0, kind="cubic")
        return torch.tensor(f(t_out), dtype=torch.float32)  # (n_pts, 3)

    refs_raw = {
        "foot" : extraer_interpolar("foot_r"),
        "knee" : extraer_interpolar("knee_r"),
        "hip"  : extraer_interpolar("hip_r"),
    }

    refs_norm, stats = {}, {}
    for nombre, t in refs_raw.items():
        mins,  _ = t.min(dim=0)
        maxs,  _ = t.max(dim=0)
        scale    = (maxs - mins).clamp_min(1e-6)
        refs_norm[nombre] = (t - mins) / scale
        stats[nombre]     = (mins, scale)

    return refs_norm, refs_raw, stats


N_POINTS = 360
refs_norm, refs_raw, stats = cargar_y_normalizar(n_pts=N_POINTS)

# Ángulos del ciclo de manivela: θ ∈ [0, 2π], 360 puntos uniformes
theta = torch.linspace(0, 2 * torch.pi, N_POINTS, device=device)  # (360,)

# Mover referencias a device
for k in refs_norm:
    refs_norm[k] = refs_norm[k].to(device)

# =============================================================================
# SECCIÓN 2 — MODELO: SERIE DE FOURIER TRUNCADA
# =============================================================================

def fourier_series(theta, coeffs):
    """
    Evalúa una serie de Fourier truncada.

    Matemática
    ----------
    x(θ) = a₀  +  Σ_{k=1}^{H} [ aₖ·cos(kθ) + bₖ·sin(kθ) ]

    El vector de coeficientes tiene la forma:
        coeffs = [ a₀,  a₁, b₁,  a₂, b₂, ...,  aH, bH ]
        largo   =  1  +  2·H

    Parámetros
    ----------
    theta  : Tensor (N,)        — ángulos en radianes
    coeffs : Tensor (1+2H,)     — coeficientes de la serie

    Retorna
    -------
    Tensor (N,) — señal reconstruida
    """
    if coeffs.dim() == 1:
        coeffs = coeffs.unsqueeze(0)          # → (1, 1+2H)

    H   = (coeffs.shape[-1] - 1) // 2        # número de armónicos

    # Componente DC (media) — k=0
    # Shape: (1, 1) × (1, N) → (1, N)
    out = coeffs[:, 0].unsqueeze(-1).expand(-1, theta.shape[0])

    # Armónicos k = 1, 2, …, H
    for k in range(1, H + 1):
        aK = coeffs[:, 2*k - 1].unsqueeze(-1)   # amplitud coseno del k-ésimo armónico
        bK = coeffs[:, 2*k    ].unsqueeze(-1)   # amplitud seno   del k-ésimo armónico

        # Suma de componentes par (coseno) e impar (seno)
        out = out + aK * torch.cos(k * theta.unsqueeze(0)) \
                  + bK * torch.sin(k * theta.unsqueeze(0))

    return out.squeeze(0)   # → (N,)


def generar_trayectorias(params, H):
    """
    Reconstruye las trayectorias 3D de foot, knee, hip evaluando la serie
    de Fourier para cada eje de cada joint.

    El vector 'params' es plano y se reorganiza como:
        params  →  (3 joints, 3 ejes, 1+2H coeficientes)

    Orden de joints: [foot, knee, hip]
    Orden de ejes:   [X, Y, Z]

    Parámetros
    ----------
    params : Tensor (dim,)    dim = 3 × 3 × (1+2H)
    H      : int              número de armónicos

    Retorna
    -------
    foot, knee, hip : Tensor (N, 3) cada una
    """
    n_per_axis = 1 + 2 * H

    # Rellenar con ceros si faltan parámetros (por robustez)
    expected = 3 * 3 * n_per_axis
    if params.numel() < expected:
        pad    = torch.zeros(expected - params.numel(), device=params.device)
        params = torch.cat([params, pad])

    p = params.view(3, 3, n_per_axis)   # (joint, eje, coef)

    # Evaluar serie de Fourier para cada eje de cada joint
    trajs = []
    for j in range(3):                  # joints: foot(0), knee(1), hip(2)
        ejes = [fourier_series(theta, p[j, ax, :]) for ax in range(3)]  # X, Y, Z
        trajs.append(torch.stack(ejes, dim=1))   # (N, 3)

    return trajs[0], trajs[1], trajs[2]   # foot, knee, hip

# =============================================================================
# SECCIÓN 3 — FUNCIÓN DE PÉRDIDA
# =============================================================================

# Pesos por joint y por eje (priorizar X y Z sobre Y lateral)
W_JOINT = torch.tensor([0.4, 0.3, 0.3], device=device)   # foot, knee, hip
W_EJE   = torch.tensor([1.0, 0.2, 1.0], device=device)   # X, Y, Z

def calcular_perdida(params, H, lambda_smooth=1e-3, lambda_l2=1e-4):
    """
    Pérdida total = MSE ponderado + penalización suavidad + regularización L2

    Términos
    --------
    1. MSE ponderado:
       Compara la trayectoria generada por Fourier contra la referencia
       de OpenSim.  Los ejes X y Z pesan más que Y (lateral).

    2. Penalización de suavidad (segunda derivada discreta):
       Penaliza cambios bruscos de aceleración: Δ²x[n] = x[n-1] - 2·x[n] + x[n+1]
       Esto actúa como un filtro pasa-bajos implícito.

    3. Regularización L2 sobre los coeficientes armónicos (excluyendo DC):
       Evita amplitudes exageradas en armónicos de alta frecuencia.
       Favorece soluciones con menos energía en frecuencias altas.
    """
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32, device=device)
    params = params.to(device)

    foot_sim, knee_sim, hip_sim = generar_trayectorias(params, H)

    # -- Normalizar simuladas igual que las referencias --
    def norm(traj, joint):
        mins, scale = stats[joint]
        return (traj - mins.to(device)) / scale.to(device)

    sims_n = [norm(foot_sim, "foot"),
              norm(knee_sim, "knee"),
              norm(hip_sim,  "hip")]
    refs_n = [refs_norm["foot"], refs_norm["knee"], refs_norm["hip"]]

    # -- Término 1: MSE ponderado por joint y por eje --
    mse_total = torch.tensor(0.0, device=device)
    for i, (s, r) in enumerate(zip(sims_n, refs_n)):
        err    = (s - r) ** 2          # (N, 3)
        mse_eje = (err * W_EJE).mean() # escalar
        mse_total = mse_total + W_JOINT[i] * mse_eje

    # -- Término 2: suavidad (segunda derivada discreta) en X y Z --
    def smooth(traj):
        # Δ²x[n] = x[n-1] - 2·x[n] + x[n+1]
        d2 = traj[:-2, :] - 2 * traj[1:-1, :] + traj[2:, :]
        return (d2[:, [0, 2]] ** 2).mean()   # solo ejes X(0) y Z(2)

    pen_smooth = smooth(foot_sim) + smooth(knee_sim) + smooth(hip_sim)

    # -- Término 3: regularización L2 sobre coeficientes armónicos --
    n_per_axis = 1 + 2 * H
    coefs = params.view(-1, n_per_axis)
    l2 = (coefs[:, 1:] ** 2).mean()           # excluir DC (columna 0)

    return mse_total + lambda_smooth * pen_smooth + lambda_l2 * l2

# =============================================================================
# SECCIÓN 4 — OPTIMIZACIÓN GLOBAL: DIFFERENTIAL EVOLUTION
# =============================================================================

def differential_evolution(dim, bounds, fn_objetivo,
                            pop_size=60, F=0.7, CR=0.9,
                            max_gens=200, seed=42):
    """
    Differential Evolution – Storn & Price (1997).

    Estrategia DE/rand/1/bin:
      mutante[i] = a + F·(b - c)      ← perturbación aleatoria de 3 individuos
      trial[i]   = mezcla(mutante, pop[i])  ← crossover binomial con prob CR
      si f(trial) < f(pop[i]) → reemplazar  ← selección greedy

    Esta estrategia busca globalmente sin calcular gradientes, lo cual es
    útil cuando la función objetivo es no convexa o tiene muchos mínimos locales.

    Parámetros
    ----------
    dim       : dimensionalidad del espacio de búsqueda
    bounds    : lista de (min, max) por dimensión
    fn_objetivo: función escalar a minimizar
    pop_size  : tamaño de la población
    F         : factor de mutación  (0 < F ≤ 2, típico 0.5–0.9)
    CR        : probabilidad de crossover  (0 ≤ CR ≤ 1, típico 0.7–0.9)
    max_gens  : generaciones máximas

    Retorna
    -------
    best_params : np.ndarray (dim,)
    best_loss   : float
    historia    : list[float]  — mejor loss por generación
    """
    np.random.seed(seed)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # Población inicial uniforme en [lb, ub]
    pop     = lb + (ub - lb) * np.random.rand(pop_size, dim)
    fitness = np.array([fn_objetivo(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best     = pop[best_idx].copy()
    best_f   = float(fitness[best_idx])
    historia = [best_f]

    print(f"  DE inicio  | loss = {best_f:.6f}  | dim = {dim}")

    for gen in range(max_gens):
        for i in range(pop_size):
            # Seleccionar 3 índices distintos y distintos de i
            candidatos = [j for j in range(pop_size) if j != i]
            a_idx, b_idx, c_idx = np.random.choice(candidatos, 3, replace=False)
            a, b, c = pop[a_idx], pop[b_idx], pop[c_idx]

            # Mutación DE/rand/1
            mutante = np.clip(a + F * (b - c), lb, ub)

            # Crossover binomial
            mascara  = np.random.rand(dim) < CR
            mascara[np.random.randint(dim)] = True      # al menos 1 gen del mutante
            trial    = np.where(mascara, mutante, pop[i])

            # Selección greedy
            f_trial = fn_objetivo(trial)
            if f_trial <= fitness[i]:
                pop[i]     = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best   = trial.copy()

        historia.append(best_f)
        if (gen + 1) % 50 == 0:
            print(f"  DE gen {gen+1:3d} | loss = {best_f:.6f}")

    return best, best_f, historia

# =============================================================================
# SECCIÓN 5 — REFINAMIENTO LOCAL: ADAM (autograd de PyTorch)
# =============================================================================

def refinar_con_adam(params_np, H, lr=3e-3, steps=400):
    """
    Refina la solución encontrada por DE usando gradiente descendente (Adam).

    Adam funciona bien aquí porque:
    - La función de pérdida ya es diferenciable (PyTorch autograd)
    - El punto inicial (salida de DE) está cerca de un mínimo local
    - Adam adapta el lr por parámetro, convergiendo más rápido que SGD

    Parámetros
    ----------
    params_np : np.ndarray  — punto inicial (salida de DE)
    H         : int         — número de armónicos
    lr        : float       — tasa de aprendizaje inicial
    steps     : int         — pasos de optimización

    Retorna
    -------
    params_np refinados : np.ndarray
    """
    params = torch.tensor(params_np, dtype=torch.float32,
                          device=device, requires_grad=True)
    opt    = torch.optim.Adam([params], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    loss_inicial = float(calcular_perdida(params.detach(), H))
    print(f"  Adam inicio | loss = {loss_inicial:.6f}")

    for t in range(steps):
        opt.zero_grad()
        loss = calcular_perdida(params, H)

        if torch.isnan(loss) or torch.isinf(loss):
            break

        loss.backward()
        opt.step()

        # Clamp suave para evitar coeficientes explosivos
        with torch.no_grad():
            params.clamp_(-2.5, 2.5)

        if (t + 1) % 100 == 0:
            print(f"  Adam  {t+1:4d}  | loss = {float(loss):.6f}")

    loss_final = float(calcular_perdida(params.detach(), H))
    print(f"  Adam fin    | loss = {loss_inicial:.6f} → {loss_final:.6f}")
    return params.detach().cpu().numpy()

# =============================================================================
# SECCIÓN 6 — PIPELINE COMPLETO
# =============================================================================

# Configuraciones a probar: (n_barras, H)
# n_barras no se usa en el cálculo Fourier, pero documenta para qué mecanismo
# se pensó esta aproximación.  H controla la riqueza espectral.
CONFIGURACIONES = [
    (6,  4),
    (7,  4),
    (8,  5),
    (12, 6),
    (16, 7),
    (20, 8),
]

resultados = []
t_inicio   = time.time()

for n_barras, H in CONFIGURACIONES:
    print(f"\n{'='*60}")
    print(f"n_barras = {n_barras}  |  H = {H}  |  dim = {3 * 3 * (1 + 2*H)}")
    print(f"{'='*60}")

    dim    = 3 * 3 * (1 + 2 * H)
    bounds = [(-2.5, 2.5)] * dim

    def obj(x):
        with torch.no_grad():
            return float(calcular_perdida(x, H))

    # Paso 1: búsqueda global con DE
    best_params, loss_de, historia = differential_evolution(
        dim, bounds, obj,
        pop_size=60, F=0.7, CR=0.9, max_gens=200
    )

    # Paso 2: refinamiento local con Adam
    best_params = refinar_con_adam(best_params, H, lr=3e-3, steps=400)
    loss_final  = obj(best_params)

    resultados.append({
        "n_barras"     : n_barras,
        "H"            : H,
        "best_params"  : best_params,      # np.ndarray (dim,)
        "final_loss"   : loss_final,
        "de_loss"      : loss_de,
        "historia_de"  : historia,
        "target_reached": loss_final < 2e-4,
    })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# SECCIÓN 7 — VISUALIZACIÓN DE LA CONVERGENCIA DE DE
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Convergencia Differential Evolution por configuración", fontsize=14)

for ax, res in zip(axes.flat, resultados):
    ax.semilogy(res["historia_de"], linewidth=1.5)
    ax.set_title(f"n_barras={res['n_barras']}, H={res['H']} | "
                 f"loss={res['final_loss']:.5f}", fontsize=9)
    ax.set_xlabel("Generación"); ax.set_ylabel("Loss (log)")
    ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("02_convergencia_de.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figura guardada: 02_convergencia_de.png")

# Guardar todos los resultados para el script 3
torch.save(resultados, "fourier_results.pt")
print("Resultados guardados: fourier_results.pt")

# Resumen final
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
mejor = min(resultados, key=lambda r: r["final_loss"])
for r in sorted(resultados, key=lambda r: r["n_barras"]):
    marca = " ← MEJOR" if r is mejor else ""
    print(f"  n_barras={r['n_barras']:2d}  H={r['H']}  "
          f"loss={r['final_loss']:.6f}{marca}")
print(f"\nTiempo total: {time.time() - t_inicio:.1f} s")
