#!/usr/bin/env python3
"""
extract_gait.py — Extrae datos de trayectoria articular de Gait2392
=====================================================================
Estrategia:
  1. Intenta usar la API de OpenSim (import opensim) para leer archivos .sto/.mot
  2. Si OpenSim no está instalado, hace parsing directo de los archivos .sto/.mot
     (que son archivos de texto tabulado con encabezado)

Salida: gait_coordinates.csv en ~/opensim_gait_data/
"""

import os
import sys
import glob
import csv

# ---------------------------------------------------------------------------
#  Configuración
# ---------------------------------------------------------------------------
WORKDIR = os.path.expanduser("~/opensim_gait_data")
OUTPUT_CSV = os.path.join(WORKDIR, "gait_coordinates.csv")

# Posibles ubicaciones de archivos de movimiento (en orden de preferencia)
SEARCH_PATHS = [
    os.path.join(WORKDIR, "Gait2392_Simbody"),
    os.path.join(WORKDIR, "Gait10dof18musc"),
    os.path.join(WORKDIR, "opensim-models", "Pipelines", "Gait2392_Simbody"),
    os.path.join(WORKDIR, "opensim-models", "Models", "Gait2392_Simbody"),
    os.path.join(WORKDIR, "opensim-models", "Models", "Gait10dof18musc"),
    os.path.join(WORKDIR, "opensim-models"),
]

# Coordenadas articulares de interés (subconjunto de columnas si existen)
JOINT_KEYWORDS = [
    "hip_flexion", "hip_adduction", "hip_rotation",
    "knee_angle", "knee",
    "ankle_angle", "ankle",
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "subtalar_angle",
]


def find_motion_files():
    """Busca archivos .sto y .mot en las rutas conocidas."""
    files = {"sto": [], "mot": []}
    for base in SEARCH_PATHS:
        if not os.path.isdir(base):
            continue
        for ext in ("sto", "mot"):
            found = glob.glob(os.path.join(base, "**", f"*.{ext}"), recursive=True)
            files[ext].extend(found)
    # Eliminar duplicados manteniendo orden
    for ext in files:
        seen = set()
        unique = []
        for f in files[ext]:
            rp = os.path.realpath(f)
            if rp not in seen:
                seen.add(rp)
                unique.append(f)
        files[ext] = unique
    return files


def parse_sto_mot_file(filepath):
    """
    Lee un archivo .sto o .mot (formato OpenSim Storage/Motion).
    Retorna (column_names, data_rows) donde data_rows es lista de listas.
    """
    column_names = []
    data_rows = []
    in_header = True
    endheader_found = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detectar fin de encabezado
            if line.lower() == "endheader":
                endheader_found = True
                in_header = False
                continue

            if in_header:
                # Algunas líneas del encabezado contienen metadata (nRows, nColumns, etc.)
                # Si no se ha encontrado "endheader", buscamos la línea de columnas
                if not endheader_found:
                    # Intentar detectar línea de columnas: empieza con "time" normalmente
                    parts = line.split("\t")
                    if len(parts) == 0:
                        parts = line.split()
                    if len(parts) > 1 and parts[0].lower() == "time":
                        column_names = parts
                        in_header = False
                    continue
                continue

            # Si ya tenemos columnas, leer datos
            if column_names:
                parts = line.split("\t")
                if len(parts) == 1:
                    parts = line.split()
                if len(parts) >= len(column_names):
                    try:
                        row = [float(x) for x in parts[:len(column_names)]]
                        data_rows.append(row)
                    except ValueError:
                        # Podría ser la línea de columnas si endheader se encontró
                        if not data_rows and parts[0].lower() == "time":
                            column_names = parts
                        continue
            else:
                # Buscar línea de columnas después de endheader
                parts = line.split("\t")
                if len(parts) == 1:
                    parts = line.split()
                if len(parts) > 1:
                    try:
                        float(parts[0])
                        # Es una fila numérica pero no tenemos columnas — raro
                        continue
                    except ValueError:
                        column_names = parts
                        continue

    return column_names, data_rows


def filter_joint_columns(column_names, data_rows):
    """Filtra columnas relevantes a articulaciones. Siempre incluye 'time'."""
    if not column_names:
        return column_names, data_rows

    # Siempre incluir time + columnas que coincidan con keywords
    indices = []
    selected_names = []
    for i, col in enumerate(column_names):
        col_lower = col.lower()
        if col_lower == "time":
            indices.append(i)
            selected_names.append(col)
        else:
            for kw in JOINT_KEYWORDS:
                if kw in col_lower:
                    indices.append(i)
                    selected_names.append(col)
                    break

    # Si no se encontraron articulaciones, devolver todas las columnas
    if len(selected_names) <= 1:
        return column_names, data_rows

    filtered_rows = [[row[i] for i in indices] for row in data_rows]
    return selected_names, filtered_rows


# ---------------------------------------------------------------------------
#  Método 1: Usar API de OpenSim
# ---------------------------------------------------------------------------
def extract_with_opensim_api():
    """Intenta usar import opensim para leer datos."""
    try:
        import opensim as osim
        print(f"[OK] OpenSim importado (versión: {getattr(osim, '__version__', 'N/A')})")
    except ImportError:
        try:
            import pyopensim as osim
            print("[OK] pyopensim importado como alternativa")
        except ImportError:
            return False

    # Buscar archivo .mot o .sto con datos de cinemática
    motion_files = find_motion_files()
    all_files = motion_files["mot"] + motion_files["sto"]

    if not all_files:
        print("[WARN] No se encontraron archivos .mot/.sto")
        return False

    target = all_files[0]
    print(f"\n[INFO] Leyendo con API OpenSim: {target}")

    try:
        table = osim.TimeSeriesTable(target)
        col_labels = list(table.getColumnLabels())

        print(f"\n{'='*60}")
        print(f"Columnas disponibles ({len(col_labels)}):")
        print(f"{'='*60}")
        for i, col in enumerate(col_labels):
            category = categorize_column(col)
            print(f"  {i+1:3d}. {col:<40s} [{category}]")

        # Exportar a CSV
        num_rows = table.getNumRows()
        with open(OUTPUT_CSV, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["time"] + col_labels)
            for i in range(num_rows):
                time_val = table.getIndependentColumn()[i]
                row_data = [time_val]
                row_vec = table.getRowAtIndex(i)
                for j in range(len(col_labels)):
                    row_data.append(row_vec[j])
                writer.writerow(row_data)

        print(f"\n[OK] Exportado {num_rows} filas × {len(col_labels)+1} columnas a:")
        print(f"     {OUTPUT_CSV}")
        return True

    except Exception as e:
        print(f"[WARN] Error usando API OpenSim: {e}")
        return False


# ---------------------------------------------------------------------------
#  Método 2: Parsing directo (fallback robusto)
# ---------------------------------------------------------------------------
def extract_with_direct_parsing():
    """Lee archivos .sto/.mot con parsing directo (sin dependencia de opensim)."""
    motion_files = find_motion_files()
    all_files = motion_files["mot"] + motion_files["sto"]

    if not all_files:
        print("[ERROR] No se encontraron archivos .mot/.sto en:")
        for p in SEARCH_PATHS:
            print(f"  - {p}")
        print("\nEjecuta primero: ./setup_opensim.sh")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Archivos de movimiento encontrados:")
    print(f"{'='*60}")
    for ext in ("mot", "sto"):
        if motion_files[ext]:
            print(f"\n  .{ext} files:")
            for f in motion_files[ext]:
                print(f"    • {os.path.relpath(f, WORKDIR)}")

    # Seleccionar el archivo principal — priorizar cinemática articular
    target = None
    # Excluir archivos que son fuerzas/GRF (no son coordenadas articulares)
    EXCLUDE_KEYWORDS = ["grf", "force", "torque", "contact", "power", "speed",
                        "activation", "control", "pErr", "pseudo", "states",
                        "BodyKinematics", "Actuation", "metabolic"]
    # Prioridad 1: Kinematics_q (coordenadas articulares generalizadas)
    # Prioridad 2: archivos IK (inverse kinematics)
    # Prioridad 3: archivos con "walk" o "gait" que no sean GRF
    priority_patterns = [
        ["Kinematics_q"],
        ["_ik", "inverse_kinematics"],
        ["walk", "gait", "normal", "subject"],
    ]
    for patterns in priority_patterns:
        for f in all_files:
            bn = os.path.basename(f).lower()
            # Skip excluded files
            if any(ex.lower() in bn for ex in EXCLUDE_KEYWORDS):
                continue
            if any(p in bn for p in patterns):
                target = f
                break
        if target:
            break
    # Fallback: first non-excluded file
    if not target:
        for f in all_files:
            bn = os.path.basename(f).lower()
            if not any(ex.lower() in bn for ex in EXCLUDE_KEYWORDS):
                target = f
                break
    if not target:
        target = all_files[0]

    print(f"\n[INFO] Archivo seleccionado: {os.path.relpath(target, WORKDIR)}")
    print(f"[INFO] Parsing directo (sin API OpenSim)...")

    col_names, data_rows = parse_sto_mot_file(target)

    if not col_names:
        print(f"[ERROR] No se pudieron leer columnas de {target}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Columnas disponibles ({len(col_names)}):")
    print(f"{'='*60}")
    for i, col in enumerate(col_names):
        category = categorize_column(col)
        print(f"  {i+1:3d}. {col:<40s} [{category}]")

    # Filtrar columnas articulares
    filtered_names, filtered_rows = filter_joint_columns(col_names, data_rows)

    print(f"\n{'='*60}")
    print(f"Columnas seleccionadas para exportación ({len(filtered_names)}):")
    print(f"{'='*60}")
    for i, col in enumerate(filtered_names):
        print(f"  {i+1:3d}. {col}")

    # Escribir CSV
    with open(OUTPUT_CSV, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(filtered_names)
        writer.writerows(filtered_rows)

    print(f"\n[OK] Exportado {len(filtered_rows)} filas × {len(filtered_names)} columnas")
    print(f"     → {OUTPUT_CSV}")

    # También exportar TODOS los archivos .sto/.mot encontrados como CSVs individuales
    export_dir = os.path.join(WORKDIR, "all_exported_csv")
    os.makedirs(export_dir, exist_ok=True)
    exported_count = 0
    for f in all_files:
        try:
            cn, dr = parse_sto_mot_file(f)
            if cn and dr:
                basename = os.path.splitext(os.path.basename(f))[0]
                out_path = os.path.join(export_dir, f"{basename}.csv")
                with open(out_path, "w", newline="") as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(cn)
                    writer.writerows(dr)
                exported_count += 1
        except Exception:
            continue

    if exported_count > 0:
        print(f"\n[INFO] También se exportaron {exported_count} archivos adicionales a:")
        print(f"       {export_dir}/")

    return True


def categorize_column(col_name):
    """Clasifica una columna por articulación/tipo."""
    cl = col_name.lower()
    if "time" in cl:
        return "Tiempo"
    if "hip" in cl:
        return "Cadera"
    if "knee" in cl:
        return "Rodilla"
    if "ankle" in cl:
        return "Tobillo"
    if "subtalar" in cl:
        return "Subtalar"
    if "pelvis" in cl:
        return "Pelvis"
    if "lumbar" in cl:
        return "Lumbar"
    if "mtp" in cl or "toe" in cl:
        return "Pie"
    if "muscle" in cl or "force" in cl:
        return "Fuerza/Músculo"
    if "ground" in cl or "contact" in cl:
        return "Contacto"
    return "Otro"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  OpenSim Gait2392 — Extracción de datos articulares")
    print("=" * 60)
    print(f"Directorio de trabajo: {WORKDIR}")
    print()

    # Intentar con API de OpenSim primero
    if not extract_with_opensim_api():
        print("\n[INFO] API OpenSim no disponible, usando parsing directo...")
        extract_with_direct_parsing()

    print(f"\n{'='*60}")
    print("  ¡Extracción completada!")
    print(f"{'='*60}")
    print(f"\nArchivo principal: {OUTPUT_CSV}")
    print(f"\nPara usar en tu proyecto de optimización:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{OUTPUT_CSV}')")
    print(f"  print(df.head())")
    print()


if __name__ == "__main__":
    main()
