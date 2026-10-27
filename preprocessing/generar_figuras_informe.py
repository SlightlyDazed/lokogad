#!/usr/bin/env python3
"""
Genera todas las figuras para el INFORME_PROCESO del Proyecto Theo Jansen GAD.
Guarda los PNGs en: prepross/figures/
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(BASE, "results")
OUT    = os.path.join(BASE, "prepross", "figures")
os.makedirs(OUT, exist_ok=True)

# ─── Paleta ───────────────────────────────────────────────────────────────────
C1, C2, C3, C4 = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
DARK = "#1C1C2E"
LIGHT = "#F0F4FF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": LIGHT,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

# ─── Datos ────────────────────────────────────────────────────────────────────
with open(os.path.join(RESULT, "evolution_history.json")) as f:
    hist = json.load(f)

with open(os.path.join(RESULT, "best_mechanism.json")) as f:
    best = json.load(f)

with open(os.path.join(RESULT, "z_layers_fixed.json")) as f:
    zlayers = json.load(f)

F_pareto = np.load(os.path.join(RESULT, "pareto_F.npy"))
top_F    = np.load(os.path.join(RESULT, "top_F_exact.npy"))
top_MN   = np.load(os.path.join(RESULT, "top_MN.npy"))

gens        = [h["gen"]          for h in hist]
best_fit    = [h["best_fitness"] for h in hist]
foot_err    = [h["best_foot"]    for h in hist]
knee_err    = [h["best_knee"]    for h in hist]
n_valid     = [h["n_valid"]      for h in hist]
best_stride = [h["best_stride"]  for h in hist]

MN_OPT = best["magic_numbers"]
MN_ORI = best["magic_numbers_original"]
LABELS  = [f"MN{i+1}" for i in range(13)]


# =============================================================================
# FIGURA 1 — Curva de Convergencia del NSGA-II (4 paneles)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Figura 1: Convergencia del Algoritmo NSGA-II\n500 Generaciones | Población = 10,000", 
             fontsize=14, fontweight='bold', y=1.01)

ax = axes[0, 0]
ax.plot(gens, best_fit, color=C1, lw=2.2, label="Fitness Global")
ax.axhline(0.11177, color=C1, ls='--', lw=1, alpha=0.6, label="Final: 0.11177")
for g, thresh, label in [(5, None, "Gen 5"), (23, None, "Gen 23"), (48, None, "Gen 48")]:
    ax.axvline(g, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(g+4, 0.30, label, fontsize=8, color='gray', rotation=90)
ax.set_title("Fitness Combinado Global", fontweight='bold')
ax.set_xlabel("Generación"); ax.set_ylabel("Fitness")
ax.legend(fontsize=9); ax.set_ylim(0.1, 0.36)

ax = axes[0, 1]
ax.plot(gens, foot_err, color=C2, lw=2.2, label="Error Pie")
ax.plot(gens, knee_err, color=C3, lw=2.2, label="Error Rodilla")
ax.axhline(0.04974, color=C2, ls='--', lw=1, alpha=0.6, label="Pie final: 4.97%")
ax.axhline(0.10791, color=C3, ls='--', lw=1, alpha=0.6, label="Rodilla final: 10.79%")
ax.set_title("Errores de Procrustes por Articulación", fontweight='bold')
ax.set_xlabel("Generación"); ax.set_ylabel("RMSD Procrustes")
ax.legend(fontsize=9)

ax = axes[1, 0]
ax.fill_between(gens, n_valid, alpha=0.3, color=C4)
ax.plot(gens, n_valid, color=C4, lw=2.2)
ax.axhline(2422, color='gray', ls=':', lw=1, alpha=0.7, label="Gen 1: 2,422 válidas")
ax.axhline(3941, color=C4, ls='--', lw=1, alpha=0.7, label="Gen 500: 3,941 válidas")
ax.set_title("Soluciones Físicamente Válidas por Generación\n(sin NaN en los 360°)", fontweight='bold')
ax.set_xlabel("Generación"); ax.set_ylabel("Nº de soluciones válidas (de 10,000)")
ax.legend(fontsize=9); ax.set_ylim(2000, 5000)

ax = axes[1, 1]
ax.plot(gens, best_stride, color="#8B5CF6", lw=2.2)
ax.axhline(966.35, color="#8B5CF6", ls='--', lw=1, alpha=0.7, label="Final: 966.35 mm")
ax.set_title("Longitud de Zancada del Mejor Individuo", fontweight='bold')
ax.set_xlabel("Generación"); ax.set_ylabel("Stride (mm)")
ax.legend(fontsize=9)

fig.tight_layout()
path = os.path.join(OUT, "fig1_convergencia_nsga2.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Guardada: {path}")


# =============================================================================
# FIGURA 2 — Magic Numbers: Comparación Original vs Optimizado
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figura 2: Genoma Optimizado — Los 13 Magic Numbers", 
             fontsize=14, fontweight='bold')

# Panel izquierdo: barras comparativas (normalizadas)
ax = axes[0]
mn_opt_norm = np.array(MN_OPT) / max(MN_OPT)
mn_ori_norm = np.array(MN_ORI) / max(MN_ORI)
x = np.arange(13)
w = 0.38
bars1 = ax.bar(x - w/2, mn_ori_norm, w, label="Original Jansen", color=C1, alpha=0.8, edgecolor='white')
bars2 = ax.bar(x + w/2, mn_opt_norm, w, label="Optimizado NSGA-II", color=C2, alpha=0.9, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=9)
ax.set_title("Comparación Original vs Optimizado\n(Normalizado al máximo de cada set)", fontweight='bold')
ax.set_ylabel("Longitud Normalizada"); ax.legend()

# Panel derecho: radar/dispersión de factores de escala
ax = axes[1]
scale = best["scale_vs_original"]
colors_bar = [C1 if s < 10 else C2 if s < 20 else C4 for s in scale]
bars = ax.bar(LABELS, scale, color=colors_bar, edgecolor='white', alpha=0.9)
ax.axhline(np.mean(scale), color='black', ls='--', lw=1.5, label=f"Media: ×{np.mean(scale):.1f}")
ax.set_title("Factor de Escala vs Jansen Original\n(Cuántas veces mayor es cada barra optimizada)", fontweight='bold')
ax.set_ylabel("Factor de Escala (×)"); ax.legend()
ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=9)
for bar, val in zip(bars, scale):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f"×{val:.1f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

fig.tight_layout()
path = os.path.join(OUT, "fig2_magic_numbers.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Guardada: {path}")


# =============================================================================
# FIGURA 3 — Frente de Pareto Top-50 (scatter 2D)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figura 3: Frente de Pareto — Top 50 Soluciones No Dominadas\n(Cinemática Exacta Re-evaluada)", 
             fontsize=14, fontweight='bold')

foot_t = top_F[:, 0]
knee_t = top_F[:, 1]
pos_t  = top_F[:, 2]

ax = axes[0]
sc = ax.scatter(foot_t * 100, knee_t * 100, c=pos_t, cmap='viridis_r', s=90, 
                edgecolors='white', lw=0.5, zorder=5)
ax.scatter(foot_t[0]*100, knee_t[0]*100, s=200, c='red', marker='*', zorder=10, label="Ganador (mejor pie)")
cb = fig.colorbar(sc, ax=ax, label="Error Posición Centroide")
ax.set_xlabel("Error Procrustes — Pie (%)")
ax.set_ylabel("Error Procrustes — Rodilla (%)")
ax.set_title("Compromiso Pie vs Rodilla\n(Color = Error de Posición Absoluta)", fontweight='bold')
ax.legend(fontsize=9)
ax.annotate(f" Ganador\n Pie: {foot_t[0]*100:.2f}%\n Rod: {knee_t[0]*100:.1f}%",
            xy=(foot_t[0]*100, knee_t[0]*100), xytext=(foot_t[0]*100+0.3, knee_t[0]*100+3),
            fontsize=8.5, arrowprops=dict(arrowstyle='->', color='red'), color='red')

ax = axes[1]
rank = np.arange(1, 51)
ax.bar(rank, foot_t*100, color=C1, alpha=0.85, label="Error Pie (%)")
ax.bar(rank, -(knee_t-knee_t.max())*100*0.3, bottom=0, color=C3, alpha=0.5, label="Error Rodilla (relativo)")
ax.axhline(foot_t[0]*100, color='red', ls='--', lw=1.5, label=f"Mínimo pie: {foot_t[0]*100:.2f}%")
ax.set_xlabel("Ranking de Solución (Top-50)")
ax.set_ylabel("Error Procrustes Pie (%)")
ax.set_title("Distribución del Error de Pie\nen el Top-50 del Frente de Pareto", fontweight='bold')
ax.legend(fontsize=9)

fig.tight_layout()
path = os.path.join(OUT, "fig3_pareto_front.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Guardada: {path}")


# =============================================================================
# FIGURA 4 — Z-Layers del Mecanismo Optimizado
# =============================================================================
z_layers = zlayers["z_layers"]
conflicts = zlayers["geo_conflicts"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figura 4: Asignación de Capas Físicas (Z-Layers)\nColoración de Grafos para Evitar Colisiones", 
             fontsize=14, fontweight='bold')

# Panel izquierdo: barras de nivel Z
ax = axes[0]
n_bars = len(z_layers)
layer_vals = np.array(z_layers)
layer_colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(set(layer_vals))))
color_map = {lv: layer_colors[i] for i, lv in enumerate(sorted(set(layer_vals)))}

for i, (lv) in enumerate(layer_vals):
    ax.barh(i+1, lv, color=color_map[lv], edgecolor='white', height=0.65, alpha=0.9)
    ax.text(lv + (0.08 if lv >= 0 else -0.08), i+1, f"Z={lv*6.6:+.1f}mm",
            va='center', ha='left' if lv >= 0 else 'right', fontsize=8.5, fontweight='bold')

ax.axvline(0, color='black', lw=1.2)
ax.set_yticks(range(1, n_bars+1))
ax.set_yticklabels([f"Barra {i+1}" for i in range(n_bars)], fontsize=9)
ax.set_xlabel("Nivel Z (pitch ≈ BAR_THICK × 2.2)")
ax.set_title("Nivel de Capa por Barra\n(0 = capa base, positivo = arriba, negativo = abajo)", fontweight='bold')

# Panel derecho: grafo de conflictos
ax = axes[1]
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, n_bars + 0.5)
ax.set_facecolor(LIGHT)

# Dibujar nodos (barras)
x_pos = np.ones(n_bars) * 1.7
y_pos = np.arange(n_bars)

for i in range(n_bars):
    c = color_map[z_layers[i]]
    circle = plt.Circle((x_pos[i], y_pos[i]), 0.28, color=c, ec='white', lw=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x_pos[i], y_pos[i], f"B{i+1}", ha='center', va='center', fontsize=8, 
            fontweight='bold', color='white' if sum(c[:3]) < 1.5 else 'black', zorder=6)

# Dibujar aristas de conflicto
for (b1, b2) in conflicts:
    ax.plot([x_pos[b1], x_pos[b2]], [y_pos[b1], y_pos[b2]], 
            'r--', lw=1.5, alpha=0.7, zorder=3)
    mid_x = (x_pos[b1] + x_pos[b2])/2 + 0.3
    mid_y = (y_pos[b1] + y_pos[b2])/2
    ax.text(mid_x, mid_y, "✗", fontsize=9, color='red', ha='center', va='center')

ax.set_xlim(0.5, 3.5)
ax.set_yticks(y_pos); ax.set_yticklabels([f"Barra {i+1}" for i in range(n_bars)], fontsize=8)
ax.set_xticks([]); ax.set_title(f"Grafo de Conflictos Cinemáticos\n({len(conflicts)} pares detectados en 360°)", fontweight='bold')

# Leyenda de colores
legend_patches = [mpatches.Patch(color=color_map[lv], label=f"Capa Z={lv}") 
                  for lv in sorted(set(z_layers))]
ax.legend(handles=legend_patches, loc='upper right', fontsize=8, title="Niveles Z")

fig.tight_layout()
path = os.path.join(OUT, "fig4_z_layers.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Guardada: {path}")


# =============================================================================
# FIGURA 5 — Resumen Ejecutivo: Panel de Métricas Finales
# =============================================================================
fig = plt.figure(figsize=(14, 8))
fig.suptitle("Figura 5: Resumen de Métricas Finales del Proyecto GAD — Theo Jansen Optimizado", 
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# ── Gauge de error de pie ──
ax1 = fig.add_subplot(gs[0, 0])
categories = ["Error\nPie\n(4.97%)", "Error\nRodilla\n(10.79%)", "Error\nPosición\n(21.07%)", ""]
values     = [4.97, 10.79, 21.07, 63.27]
colors_pie = [C1, C2, C3, '#E0E0E0']
wedges, texts = ax1.pie(values, colors=colors_pie, startangle=90,
                        wedgeprops=dict(width=0.5, edgecolor='white'))
ax1.text(0, 0, "Fitness\n0.1118", ha='center', va='center', fontsize=11, fontweight='bold')
ax1.set_title("Distribución de Errores\ndel Individuo Ganador", fontweight='bold', fontsize=10)
ax1.legend(wedges[:3], ["Pie: 4.97%", "Rodilla: 10.79%", "Posición: 21.07%"], 
           loc='lower center', fontsize=7.5, ncol=2)

# ── KPIs en tabla ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
table_data = [
    ["Métrica",                 "Valor"],
    ["Error Procrustes Pie",    "4.97%"],
    ["Error Procrustes Rodilla","10.79%"],
    ["Longitud de Zancada",     "966 mm"],
    ["Generaciones NSGA-II",    "500"],
    ["Población por Gen.",      "10,000"],
    ["Generación de Convergencia","~Gen 23"],
    ["Soluciones Válidas (fin)","3,941 / 10K"],
    ["Capas Z (ensamble)",      "6 niveles"],
    ["Grosor axial ensamble",   "~33 mm"],
]
tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1.2, 1.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4C72B0')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#E8EFFF')
ax2.set_title("KPIs del Proyecto", fontweight='bold', fontsize=10, pad=20)

# ── Convergencia rápida ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(gens[:100], foot_err[:100], color=C2, lw=2)
ax3.fill_between(gens[:100], foot_err[:100], alpha=0.2, color=C2)
ax3.axvline(23, color='red', ls='--', lw=1.5, label="Gen 23: <0.15")
ax3.set_xlabel("Generación"); ax3.set_ylabel("Error Procrustes Pie")
ax3.set_title("Convergencia del Error de Pie\n(Primeras 100 generaciones)", fontweight='bold', fontsize=10)
ax3.legend(fontsize=9)

# ── Comparación MN clave ──
ax4 = fig.add_subplot(gs[1, :2])
x = np.arange(13)
w = 0.38
ax4.bar(x - w/2, MN_ORI, w, color=C1, alpha=0.8, label="Original Jansen (mm)", edgecolor='white')
ax4.bar(x + w/2, [m/max(MN_OPT)*max(MN_ORI)*4 for m in MN_OPT], w, 
        color=C2, alpha=0.8, label="Optimizado NSGA-II (escala ÷4 para visualizar)", edgecolor='white')
ax4.set_xticks(x); ax4.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel("Longitud (mm)"); ax4.legend(fontsize=9)
ax4.set_title("Magic Numbers: Original vs Optimizado\n(Optimizado escalado ÷4 para visualización comparativa)", 
              fontweight='bold', fontsize=10)

# ── Z-layers ──
ax5 = fig.add_subplot(gs[1, 2])
unique_layers = sorted(set(z_layers))
counts = [z_layers.count(l) for l in unique_layers]
colors_zl = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(unique_layers)))
ax5.barh([f"Z={l}" for l in unique_layers], counts, color=colors_zl, edgecolor='white', alpha=0.9)
for i, (v, l) in enumerate(zip(counts, unique_layers)):
    ax5.text(v + 0.05, i, f"{v} barra{'s' if v > 1 else ''}", va='center', fontsize=9)
ax5.set_xlabel("Nº de barras en esa capa")
ax5.set_title("Distribución de Barras\npor Nivel Z", fontweight='bold', fontsize=10)
ax5.set_xlim(0, max(counts) + 1.5)

path = os.path.join(OUT, "fig5_resumen_ejecutivo.png")
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Guardada: {path}")

print(f"\n✅ Todas las figuras guardadas en: {OUT}")
