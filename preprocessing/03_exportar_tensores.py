# =============================================================================
# 03_exportar_tensores.py
# Reconstruye y exporta las trayectorias articulares optimizadas.
#
# Lee los coeficientes Fourier guardados por 02_aproximacion_fourier.py,
# evalúa la serie para cada configuración y guarda:
#   • all_extracted_trajectories.pt  — lista con los 6 resultados
#   • foot_optimized_tensor.pt       — mejor config: pie   (360, 3)
#   • knee_optimized_tensor.pt       — mejor config: rodilla (360, 3)
#   • hip_optimized_tensor.pt        — mejor config: cadera  (360, 3)
#
# Entrada : fourier_results.pt  (salida de 02_aproximacion_fourier.py)
# =============================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt

RESULTS_FILE = "fourier_results.pt"

# -----------------------------------------------------------------------------
# 1. FUNCIONES FOURIER (misma lógica que en el script 2)
# -----------------------------------------------------------------------------

N_POINTS = 360
theta = torch.linspace(0, 2 * torch.pi, N_POINTS)   # (360,) en CPU

def fourier_series(theta, coeffs):
    """Serie de Fourier truncada: a₀ + Σ[aₖcos(kθ) + bₖsin(kθ)]"""
    if coeffs.dim() == 1:
        coeffs = coeffs.unsqueeze(0)
    H   = (coeffs.shape[-1] - 1) // 2
    out = coeffs[:, 0].unsqueeze(-1).expand(-1, theta.shape[0])
    for k in range(1, H + 1):
        aK = coeffs[:, 2*k - 1].unsqueeze(-1)
        bK = coeffs[:, 2*k    ].unsqueeze(-1)
        out = out + aK * torch.cos(k * theta.unsqueeze(0)) \
                  + bK * torch.sin(k * theta.unsqueeze(0))
    return out.squeeze(0)


def reconstruir_trayectorias(params, H):
    """
    Reconstruye foot, knee, hip (360, 3) a partir del vector de parámetros.

    params puede ser np.ndarray o torch.Tensor.
    """
    if isinstance(params, np.ndarray):
        params = torch.tensor(params, dtype=torch.float32)

    n_per_axis = 1 + 2 * H
    expected   = 3 * 3 * n_per_axis
    if params.numel() < expected:
        params = torch.cat([params, torch.zeros(expected - params.numel())])

    p = params.view(3, 3, n_per_axis)   # (joint, eje, coef)

    trajs = []
    for j in range(3):
        ejes = [fourier_series(theta, p[j, ax, :]) for ax in range(3)]
        trajs.append(torch.stack(ejes, dim=1).detach().cpu())

    return trajs[0], trajs[1], trajs[2]   # foot, knee, hip  — cada (360, 3)

# -----------------------------------------------------------------------------
# 2. CARGAR RESULTADOS Y RECONSTRUIR TRAYECTORIAS
# -----------------------------------------------------------------------------

print("Cargando resultados Fourier...")
resultados = torch.load(RESULTS_FILE, weights_only=False)
print(f"  {len(resultados)} configuraciones encontradas")

extracted = []

for res in resultados:
    n_barras    = res["n_barras"]
    H           = res["H"]
    params      = res["best_params"]
    final_loss  = res["final_loss"]
    target      = res.get("target_reached", False)

    print(f"\n  Reconstruyendo: n_barras={n_barras}  H={H}  loss={final_loss:.6f}")

    foot, knee, hip = reconstruir_trayectorias(params, H)

    print(f"    foot  {tuple(foot.shape)}  X=[{foot[:,0].min():.3f}, {foot[:,0].max():.3f}]")
    print(f"    knee  {tuple(knee.shape)}  X=[{knee[:,0].min():.3f}, {knee[:,0].max():.3f}]")
    print(f"    hip   {tuple(hip.shape)}   X=[{hip[:,0].min():.3f}, {hip[:,0].max():.3f}]")

    extracted.append({
        "index"          : len(extracted),
        "n_barras"       : n_barras,
        "H"              : H,
        "final_loss"     : final_loss,
        "target_reached" : target,
        "foot_trajectory": foot,     # (360, 3)
        "knee_trajectory": knee,     # (360, 3)
        "hip_trajectory" : hip,      # (360, 3)
        "original_params": params,   # np.ndarray — coeficientes Fourier
    })

# -----------------------------------------------------------------------------
# 3. SELECCIONAR LA MEJOR CONFIGURACIÓN
# -----------------------------------------------------------------------------

mejor = min(extracted, key=lambda x: x["final_loss"])

print(f"\n{'='*60}")
print("RESUMEN — todas las configuraciones")
print(f"{'='*60}")
for e in extracted:
    marca = " ← MEJOR" if e is mejor else ""
    print(f"  idx={e['index']}  n_barras={e['n_barras']:2d}  H={e['H']}  "
          f"loss={e['final_loss']:.6f}  target={e['target_reached']}{marca}")

print(f"\nMejor: n_barras={mejor['n_barras']}  H={mejor['H']}  "
      f"loss={mejor['final_loss']:.6f}")

# -----------------------------------------------------------------------------
# 4. GUARDAR TENSORES
# -----------------------------------------------------------------------------

# 4a. Todos los resultados (para análisis posterior)
torch.save(extracted, "all_extracted_trajectories.pt")
print("\nGuardado: all_extracted_trajectories.pt")

# 4b. Tensores individuales de la mejor configuración
torch.save(mejor["foot_trajectory"], "foot_optimized_tensor.pt")
torch.save(mejor["knee_trajectory"], "knee_optimized_tensor.pt")
torch.save(mejor["hip_trajectory"],  "hip_optimized_tensor.pt")
print("Guardado: foot_optimized_tensor.pt")
print("Guardado: knee_optimized_tensor.pt")
print("Guardado: hip_optimized_tensor.pt")

# -----------------------------------------------------------------------------
# 5. VISUALIZACIÓN: COMPARACIÓN REFERENCIA vs FOURIER (mejor config)
# -----------------------------------------------------------------------------

# Cargar referencias crudas para comparar
import scipy.io as sio
from scipy.interpolate import interp1d

mat = sio.loadmat("joint_trajectories_opensim.mat")
jt  = mat["joint_trajectories"]

def ref_interp(campo, n=360):
    arr  = jt[campo][0, 0].astype(np.float32)
    t_in = np.linspace(0, 1, arr.shape[0])
    t_out= np.linspace(0, 1, n)
    return interp1d(t_in, arr, axis=0, kind="cubic")(t_out)

foot_ref = ref_interp("foot_r")
knee_ref = ref_interp("knee_r")
hip_ref  = ref_interp("hip_r")

foot_opt = mejor["foot_trajectory"].numpy()
knee_opt = mejor["knee_trajectory"].numpy()
hip_opt  = mejor["hip_trajectory"].numpy()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    f"Referencia OpenSim vs Reconstrucción Fourier\n"
    f"(n_barras={mejor['n_barras']}, H={mejor['H']}, loss={mejor['final_loss']:.5f})",
    fontsize=13
)

theta_deg = np.linspace(0, 360, 360)

# Fila 0: plano sagital (X–Z)
for col, (nombre, ref, opt, color) in enumerate([
    ("Pie",     foot_ref, foot_opt, "tab:red"),
    ("Rodilla", knee_ref, knee_opt, "tab:green"),
    ("Cadera",  hip_ref,  hip_opt,  "tab:blue"),
]):
    ax = axes[0, col]
    ax.plot(ref[:, 0], ref[:, 2], color=color, lw=2.5, label="OpenSim")
    ax.plot(opt[:, 0], opt[:, 2], "k--",      lw=1.5, label="Fourier", alpha=0.8)
    ax.set_title(f"{nombre} — plano sagital (X–Z)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

# Fila 1: error punto a punto
for col, (nombre, ref, opt) in enumerate([
    ("Pie",     foot_ref, foot_opt),
    ("Rodilla", knee_ref, knee_opt),
    ("Cadera",  hip_ref,  hip_opt),
]):
    ax   = axes[1, col]
    err  = np.sqrt(((ref - opt) ** 2).sum(axis=1))    # error euclídeo en metros
    ax.plot(theta_deg, err * 1e3, linewidth=1.5)       # en mm
    ax.axhline(err.mean() * 1e3, color="red", linestyle="--",
               label=f"media = {err.mean()*1e3:.2f} mm")
    ax.set_title(f"Error {nombre} vs ángulo de ciclo")
    ax.set_xlabel("Ángulo (°)"); ax.set_ylabel("Error 3D (mm)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("03_comparacion_fourier.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigura guardada: 03_comparacion_fourier.png")

# -----------------------------------------------------------------------------
# 6. VERIFICACIÓN RÁPIDA DE LOS ARCHIVOS GUARDADOS
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("VERIFICACIÓN DE ARCHIVOS .pt")
print("=" * 60)

archivos = [
    "all_extracted_trajectories.pt",
    "foot_optimized_tensor.pt",
    "knee_optimized_tensor.pt",
    "hip_optimized_tensor.pt",
]

for path in archivos:
    data = torch.load(path, weights_only=False)
    if isinstance(data, torch.Tensor):
        print(f"  {path:40s}  shape={tuple(data.shape)}  "
              f"dtype={data.dtype}  "
              f"rango=[{data.min():.3f}, {data.max():.3f}]")
    elif isinstance(data, list):
        print(f"  {path:40s}  lista de {len(data)} dicts")
