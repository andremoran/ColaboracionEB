#!/usr/bin/env bash
# Evaluaciones finales del Proyecto 1 y del atlas (se lanza al terminar el entrenamiento de K3Mov v2).
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
echo "== evaluación K3Mov" && OMP_NUM_THREADS=4 $PY -u P1_K3Mov/evaluar.py --modelo P1_K3Mov/modelos/k3mov_v2/mejor.pt --n 240 --mmfit --ec3d --hilos 4
echo "== líneas base" && $PY -u P1_K3Mov/lineas_base.py --pasos_gru 1500 --hilos 3
echo "== video del decodificador" && OMP_NUM_THREADS=2 $PY -u P1_K3Mov/video_correccion.py --ejercicio sentadilla --error talones_elevados
OMP_NUM_THREADS=2 $PY -u P1_K3Mov/video_correccion.py --ejercicio sentadilla --error rodillas_valgo
echo "== atlas (reevaluación con filtrado de ruido)" && OMP_NUM_THREADS=1 $PY -u P5_AtlasAvatar3D/evaluar.py
echo "== fin de la cadena"
