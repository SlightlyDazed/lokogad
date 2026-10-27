#!/usr/bin/env bash
# =============================================================================
# setup_opensim.sh — Instalar OpenSim (Python API) en OpenSUSE sin Conda
# Ejecutar desde ~/opensim_gait_data/
# Uso:  chmod +x setup_opensim.sh && ./setup_opensim.sh
# =============================================================================
set -euo pipefail

WORKDIR="$HOME/opensim_gait_data"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "========================================"
echo " 1/5  Instalando dependencias del sistema"
echo "========================================"
sudo zypper install -y python3 python3-pip python3-devel git cmake gcc gcc-c++ \
    swig java-11-openjdk java-11-openjdk-devel \
    lapack-devel blas-devel libXmu-devel libXi-devel freeglut-devel \
    Mesa-libGL-devel || {
    echo "[WARN] Algunos paquetes no se encontraron. Revisa los nombres para tu versión de OpenSUSE."
}

echo ""
echo "========================================"
echo " 2/5  Creando entorno virtual Python"
echo "========================================"
python3 -m venv "$WORKDIR/venv"
source "$WORKDIR/venv/bin/activate"
pip install --upgrade pip

echo ""
echo "========================================"
echo " 3/5  Instalando OpenSim via pyopensim (pip)"
echo "========================================"
echo "[INFO] Intentando instalar pyopensim (paquete pip no-oficial con bindings bundled)..."
if pip install pyopensim; then
    echo "[OK] pyopensim instalado correctamente."
    INSTALL_METHOD="pyopensim"
else
    echo "[WARN] pyopensim falló. Intentando opensim..."
    if pip install opensim; then
        echo "[OK] opensim instalado correctamente."
        INSTALL_METHOD="opensim"
    else
        echo ""
        echo "============================================================"
        echo " pyopensim / opensim via pip NO están disponibles para tu"
        echo " plataforma. Se procederá a compilar desde el código fuente."
        echo " Esto puede tardar 30-60 minutos."
        echo "============================================================"
        INSTALL_METHOD="source"
    fi
fi

# --- Compilación desde fuente (solo si pip falló) ---
if [ "$INSTALL_METHOD" = "source" ]; then
    echo ""
    echo "========================================"
    echo " 3b/5  Compilando opensim-core desde fuente"
    echo "========================================"
    SRCDIR="$WORKDIR/opensim-core-src"
    BUILDDIR="$WORKDIR/opensim-core-build"
    INSTALLDIR="$WORKDIR/opensim-core-install"

    if [ ! -d "$SRCDIR" ]; then
        git clone https://github.com/opensim-org/opensim-core.git "$SRCDIR"
    fi

    mkdir -p "$BUILDDIR"
    cd "$BUILDDIR"

    cmake "$SRCDIR" \
        -DCMAKE_INSTALL_PREFIX="$INSTALLDIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_WRAPPING=ON \
        -DBUILD_JAVA_WRAPPING=OFF \
        -DOPENSIM_PYTHON_VERSION=3 \
        -DBUILD_TESTING=OFF \
        -DOPENSIM_WITH_CASADI=OFF \
        -DOPENSIM_WITH_TROPTER=OFF

    make -j"$(nproc)"
    make install

    # Configurar PYTHONPATH
    PYTHON_SITE=$(find "$INSTALLDIR" -type d -name "opensim" | head -1 | xargs dirname)
    if [ -n "$PYTHON_SITE" ]; then
        export PYTHONPATH="$PYTHON_SITE:${PYTHONPATH:-}"
        echo "export PYTHONPATH=\"$PYTHON_SITE:\${PYTHONPATH:-}\"" >> "$WORKDIR/venv/bin/activate"
    fi

    # Configurar LD_LIBRARY_PATH
    LIBDIR="$INSTALLDIR/lib"
    export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"
    echo "export LD_LIBRARY_PATH=\"$LIBDIR:\${LD_LIBRARY_PATH:-}\"" >> "$WORKDIR/venv/bin/activate"

    cd "$WORKDIR"
    echo "[OK] opensim-core compilado e instalado en $INSTALLDIR"
fi

echo ""
echo "========================================"
echo " 4/5  Descargando modelos de ejemplo (Gait2392)"
echo "========================================"
MODELS_DIR="$WORKDIR/opensim-models"
if [ ! -d "$MODELS_DIR" ]; then
    git clone --depth 1 https://github.com/opensim-org/opensim-models.git "$MODELS_DIR"
else
    echo "[INFO] opensim-models ya existe, no se vuelve a clonar."
fi

# Copiar Gait2392 y Gait10dof18musc al directorio de trabajo
echo "[INFO] Copiando modelos Gait al directorio de trabajo..."
if [ -d "$MODELS_DIR/Pipelines/Gait2392_Simbody" ]; then
    cp -r "$MODELS_DIR/Pipelines/Gait2392_Simbody" "$WORKDIR/Gait2392_Simbody"
    echo "[OK] Gait2392_Simbody copiado."
elif [ -d "$MODELS_DIR/Models/Gait2392_Simbody" ]; then
    cp -r "$MODELS_DIR/Models/Gait2392_Simbody" "$WORKDIR/Gait2392_Simbody"
    echo "[OK] Gait2392_Simbody copiado."
else
    echo "[WARN] No se encontró Gait2392_Simbody. Buscando..."
    FOUND=$(find "$MODELS_DIR" -type d -iname "*gait2392*" | head -1)
    if [ -n "$FOUND" ]; then
        cp -r "$FOUND" "$WORKDIR/Gait2392_Simbody"
        echo "[OK] Encontrado y copiado: $FOUND"
    else
        echo "[ERROR] No se encontró el modelo Gait2392 en el repositorio."
    fi
fi

if [ -d "$MODELS_DIR/Models/Gait10dof18musc" ]; then
    cp -r "$MODELS_DIR/Models/Gait10dof18musc" "$WORKDIR/Gait10dof18musc"
    echo "[OK] Gait10dof18musc copiado."
fi

# Instalar dependencias Python para el script de extracción
pip install pandas numpy

echo ""
echo "========================================"
echo " 5/5  Verificando instalación"
echo "========================================"
python3 -c "
try:
    import opensim as osim
    print('[OK] import opensim exitoso — versión:', osim.__version__ if hasattr(osim, '__version__') else 'desconocida')
except ImportError:
    try:
        import pyopensim as osim
        print('[OK] import pyopensim exitoso')
    except ImportError:
        print('[WARN] No se pudo importar opensim ni pyopensim.')
        print('       El script extract_gait.py tiene un fallback que lee .sto/.mot directamente.')
"

echo ""
echo "========================================"
echo " ¡Instalación completa!"
echo "========================================"
echo ""
echo "Directorio de trabajo: $WORKDIR"
echo "Para activar el entorno: source $WORKDIR/venv/bin/activate"
echo "Para extraer datos:      python3 $WORKDIR/extract_gait.py"
echo ""
echo "Archivos .sto y .mot disponibles:"
find "$WORKDIR/Gait2392_Simbody" -name "*.sto" -o -name "*.mot" 2>/dev/null || echo "(se listarán tras la descarga)"
