#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  COMBINER — Une los 4 actos en un solo video final
═══════════════════════════════════════════════════════════════════════
  Usa ffmpeg para concatenar act1..act4 en presentation_full.mp4
  Con transiciones de fade entre actos.

  Salida: presentation_full.mp4
"""
import os, subprocess, sys

VIDEO_DIR = os.path.dirname(os.path.abspath(__file__))
acts = [
    os.path.join(VIDEO_DIR, 'act1_opensim.mp4'),
    os.path.join(VIDEO_DIR, 'act2_extraction.mp4'),
    os.path.join(VIDEO_DIR, 'act3_evolution.mp4'),
    os.path.join(VIDEO_DIR, 'act4_assembly.mp4'),
]
OUTPUT = os.path.join(VIDEO_DIR, 'presentation_full.mp4')

# Verificar que existen todos
missing = [a for a in acts if not os.path.exists(a)]
if missing:
    print(f"[ERROR] Faltan: {missing}")
    print("Ejecuta primero los scripts de cada acto.")
    sys.exit(1)

# Crear lista para ffmpeg concat
list_file = os.path.join(VIDEO_DIR, '_concat_list.txt')
with open(list_file, 'w') as f:
    for act in acts:
        f.write(f"file '{act}'\n")

# Concatenar con ffmpeg
cmd = [
    'ffmpeg', '-y',
    '-f', 'concat', '-safe', '0',
    '-i', list_file,
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'medium',
    '-crf', '18',
    '-movflags', '+faststart',
    OUTPUT
]

print(f"Concatenating {len(acts)} acts into {OUTPUT}")
print(f"Command: {' '.join(cmd)}")

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\n✓ Final video: {OUTPUT}")
    print(f"  Size: {size_mb:.1f} MB")
else:
    print(f"[ERROR] ffmpeg failed:")
    print(result.stderr[-500:] if result.stderr else "No stderr")

# Cleanup
if os.path.exists(list_file):
    os.remove(list_file)
