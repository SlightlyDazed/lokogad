#!/bin/bash
# run_jansen_pipeline.sh — Ejecuta el pipeline GAD completo para Theo Jansen.
# Uso: bash run_jansen_pipeline.sh

set -e
cd "$(dirname "$0")"
# Activa tu entorno de Python antes de ejecutar (ver README).

echo "================================================================"
echo "  GAD Theo Jansen — Pipeline Completo"
echo "================================================================"

# Fase 1: Dataset
echo -e "\n>>> FASE 1: Generando dataset..."
python3 pipeline/fase1_jansen_dataset.py --n-samples 10000 --workers 8

# Fase 2: GNN
echo -e "\n>>> FASE 2: Entrenando GNN..."
python3 pipeline/fase2_jansen_gat.py --dataset data/jansen_dataset.pt

# Fase 3: NSGA
echo -e "\n>>> FASE 3: Optimización NSGA-II..."
python3 pipeline/fase3_jansen_nsga.py --n-gen 500 --pop-size 10000

# Fase 3b: Refinamiento topológico (mecanismo final de 12 barras del paper)
echo -e "\n>>> FASE 3b: Refinamiento topológico..."
python3 pipeline/fase3_topology_evo.py --n-gen 500

# Fase 4: Export
echo -e "\n>>> FASE 4: Exportando candidatos..."
python3 pipeline/fase4_jansen_export.py --top-n 5

echo -e "\n================================================================"
echo "  Pipeline completado. Resultados en results/"
echo "================================================================"
