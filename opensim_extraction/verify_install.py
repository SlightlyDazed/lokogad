#!/usr/bin/env python3
"""
verify_install.py — Verificar que OpenSim y los datos de Gait2392 funcionan.
"""
import os
import sys

WORKDIR = os.path.expanduser("~/opensim_gait_data")
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

def check(label, condition, note=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  ({note})" if note else ""))
    return condition

def main():
    print("=" * 60)
    print("  Verificación de instalación OpenSim + Gait2392")
    print("=" * 60)
    results = []

    # 1. Directorio de trabajo
    results.append(check("Directorio ~/opensim_gait_data existe", os.path.isdir(WORKDIR)))

    # 2. Modelo Gait2392
    gait_dir = os.path.join(WORKDIR, "Gait2392_Simbody")
    results.append(check("Modelo Gait2392_Simbody copiado", os.path.isdir(gait_dir)))

    # 3. Archivos .osim
    osim_files = []
    for root, dirs, files in os.walk(WORKDIR):
        for f in files:
            if f.endswith(".osim"):
                osim_files.append(os.path.join(root, f))
    results.append(check(f"Archivos .osim encontrados: {len(osim_files)}", len(osim_files) > 0))

    # 4. Archivos .sto / .mot
    sto_files, mot_files = [], []
    for root, dirs, files in os.walk(WORKDIR):
        for f in files:
            if f.endswith(".sto"):
                sto_files.append(os.path.join(root, f))
            elif f.endswith(".mot"):
                mot_files.append(os.path.join(root, f))
    results.append(check(f"Archivos .sto encontrados: {len(sto_files)}", len(sto_files) > 0))
    results.append(check(f"Archivos .mot encontrados: {len(mot_files)}", len(mot_files) > 0))

    # 5. API Python de OpenSim
    try:
        import opensim as osim
        results.append(check("import opensim", True, f"v{getattr(osim, '__version__', '?')}"))
    except ImportError:
        try:
            import pyopensim as osim
            results.append(check("import pyopensim (alternativa)", True))
        except ImportError:
            print(f"  {WARN} import opensim/pyopensim no disponible")
            print(f"        El script extract_gait.py funcionará con parsing directo.")

    # 6. CSV de salida
    csv_path = os.path.join(WORKDIR, "gait_coordinates.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        results.append(check(f"gait_coordinates.csv: {rows} filas × {len(header)} cols", rows > 0))
    else:
        print(f"  {WARN} gait_coordinates.csv aún no generado (ejecuta extract_gait.py)")

    # Resumen
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Resultado: {passed}/{total} verificaciones pasadas")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
