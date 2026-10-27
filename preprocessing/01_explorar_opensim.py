# =============================================================================
# 01_explorar_opensim.py
# Exploración y visualización de trayectorias articulares desde OpenSim (.mat)
#
# Entrada : joint_trajectories_opensim.mat
# Salida  : prints descriptivos + figura PNG de las trayectorias crudas
# =============================================================================

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

MAT_FILE = "joint_trajectories_opensim.mat"

# -----------------------------------------------------------------------------
# 1. CARGA DEL ARCHIVO .mat
# -----------------------------------------------------------------------------
# Los archivos .mat exportados de OpenSim son structs de MATLAB.
# scipy.io.loadmat los convierte a numpy structured arrays.
mat = sio.loadmat(MAT_FILE)

# Extraer el struct principal
joint_traj = mat["joint_trajectories"]

# Listar todos los campos disponibles (joints exportados desde OpenSim)
campos = joint_traj.dtype.names
print("=" * 60)
print("JOINTS disponibles en el archivo:")
print("=" * 60)
for campo in campos:
    arr = joint_traj[campo][0, 0]          # acceso al array dentro del struct
    print(f"  {campo:>12s}  →  shape {arr.shape}  "
          f"| X=[{arr[:,0].min():.3f}, {arr[:,0].max():.3f}]  "
          f"| Z=[{arr[:,2].min():.3f}, {arr[:,2].max():.3f}]  m")

# -----------------------------------------------------------------------------
# 2. EXTRACCIÓN DE LOS JOINTS QUE NOS INTERESAN (lado derecho)
# -----------------------------------------------------------------------------
# Cada joint tiene shape (N_frames, 3) donde las columnas son X, Y, Z en metros.
# N_frames = 73 → representa un ciclo completo de marcha (0% a 100% gait cycle).
# Usamos lado derecho (_r) que es el que OpenSim exporta primero.

foot_r  = joint_traj["foot_r"][0, 0]    # (73, 3)
knee_r  = joint_traj["knee_r"][0, 0]    # (73, 3)
hip_r   = joint_traj["hip_r"][0, 0]     # (73, 3)
ankle_r = joint_traj["ankle_r"][0, 0]   # (73, 3)  — útil como referencia

N_frames = foot_r.shape[0]
print(f"\nFrames originales por ciclo de marcha: {N_frames}")

# -----------------------------------------------------------------------------
# 3. ESTADÍSTICAS BÁSICAS
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ESTADÍSTICAS (metros)")
print("=" * 60)

for nombre, arr in [("foot_r", foot_r), ("knee_r", knee_r),
                    ("hip_r",  hip_r),  ("ankle_r", ankle_r)]:
    rng_x = arr[:, 0].max() - arr[:, 0].min()
    rng_z = arr[:, 2].max() - arr[:, 2].min()
    print(f"  {nombre}  |  stride X = {rng_x:.4f} m  |  altura Z: "
          f"[{arr[:,2].min():.3f}, {arr[:,2].max():.3f}] m  |  rango Z = {rng_z:.4f} m")

# -----------------------------------------------------------------------------
# 4. INTERPOLACIÓN: 73 frames → 360 puntos
# -----------------------------------------------------------------------------
# Interpolamos a 360 puntos para que cada punto corresponda a 1° del ciclo
# de manivela (θ ∈ [0, 2π]).  Esto facilita calcular la serie de Fourier
# directamente sobre ángulos uniformes.

def interpolar_trayectoria(traj, n_salida=360):
    """Interpola cúbicamente una trayectoria (N_in, 3) → (n_salida, 3)."""
    t_in  = np.linspace(0, 1, traj.shape[0])
    t_out = np.linspace(0, 1, n_salida)
    f = interp1d(t_in, traj, axis=0, kind="cubic")
    return f(t_out)

foot_360  = interpolar_trayectoria(foot_r)
knee_360  = interpolar_trayectoria(knee_r)
hip_360   = interpolar_trayectoria(hip_r)
ankle_360 = interpolar_trayectoria(ankle_r)

print(f"\nDespués de interpolación: {foot_360.shape}  (360 × 3)")

# -----------------------------------------------------------------------------
# 5. VISUALIZACIÓN
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Trayectorias articulares – OpenSim (ciclo de marcha completo)",
             fontsize=14, fontweight="bold")

theta_deg = np.linspace(0, 360, 360)   # eje angular del ciclo

joints = [
    ("Pie",    foot_360,  "tab:red"),
    ("Rodilla", knee_360, "tab:green"),
    ("Cadera", hip_360,   "tab:blue"),
]

# Fila 0: trayectorias en el plano sagital (X–Z)
for col, (nombre, traj, color) in enumerate(joints):
    ax = axes[0, col]
    ax.plot(traj[:, 0], traj[:, 2], color=color, linewidth=2)
    # marcar inicio (verde) y fin (rojo) del ciclo
    ax.scatter(traj[0, 0],  traj[0, 2],  color="limegreen", zorder=5,
               s=60, label="inicio (0°)")
    ax.scatter(traj[-1, 0], traj[-1, 2], color="crimson",   zorder=5,
               s=60, label="fin (360°)")
    ax.set_title(f"{nombre} – plano sagital (X–Z)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

# Fila 1: evolución temporal de X y Z por separado
for col, (nombre, traj, color) in enumerate(joints):
    ax = axes[1, col]
    ax.plot(theta_deg, traj[:, 0], color=color,
            linewidth=2, label="X (avance)")
    ax.plot(theta_deg, traj[:, 2], color=color,
            linewidth=2, linestyle="--", label="Z (altura)", alpha=0.7)
    ax.set_title(f"{nombre} – X y Z vs ángulo de ciclo")
    ax.set_xlabel("Ángulo de ciclo (°)")
    ax.set_ylabel("Posición (m)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_exploracion_opensim.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigura guardada: 01_exploracion_opensim.png")
