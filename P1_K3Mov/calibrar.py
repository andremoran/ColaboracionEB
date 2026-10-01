"""Ajusta la calibración de las probabilidades de error de K3Mov con la partición de VALIDACIÓN
(sujetos que no se usaron para entrenar ni para evaluar). Ver k3mov/calibracion.py.

Uso:  python P1_K3Mov/calibrar.py [--modelo ...] [--n 240]
Salida: <carpeta del modelo>/calibracion_errores.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from evaluar import cargar, inferir, auc  # noqa: E402
from k3mov.calibracion import ajustar, aplicar, ARCHIVO  # noqa: E402
from k3mov.datos import Cache, construir_cache, armar_celdas  # noqa: E402
from nucleo.ejercicios import CATALOGO_ERRORES  # noqa: E402


def puntajes(modelo, cache, idx):
    """Puntajes y etiquetas de error por repetición (media) y por serie (percentil 90)."""
    d = cache.d
    rep_, ser_ = dict(y=[], s=[], t=[]), dict(y=[], s=[], t=[])
    for s in idx:
        i0, T = int(d["seq_inicio"][s]), int(d["seq_T"][s])
        sl = slice(i0, i0 + T)
        cel, P = armar_celdas(A=d["angulos_captura"][sl].astype(np.float64), L=d["captura_T"][sl].astype(np.float64),
                              vis=d["captura_vis"][sl].astype(np.float64), imu=d["imu"][sl].astype(np.float64),
                              sujeto=d["seq_sujeto"][s], T=T)
        p_err = inferir(modelo, cel, P, decodificar=False)["p_err"]
        err_true = d["err"][sl].astype(np.float32)
        rep = d["rep"][sl].astype(int)
        for j, g in enumerate(d["seq_err_idx"][s]):
            if g < 0:
                continue
            tipo = CATALOGO_ERRORES[g]
            for k in np.unique(rep[rep > 0]):
                m = rep == k
                rep_["y"].append(int(err_true[m, j].max() > 0.3)); rep_["s"].append(float(p_err[m, g].mean()))
                rep_["t"].append(tipo)
            ser_["y"].append(int(err_true[:, j].max() > 0.3)); ser_["s"].append(float(np.percentile(p_err[:, g], 90)))
            ser_["t"].append(tipo)
    return rep_, ser_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(AQUI / "modelos" / "k3mov_v2" / "mejor.pt"))
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    modelo = cargar(a.modelo)
    cache = Cache(construir_cache())
    _, va, _ = cache.particion()
    rep_, ser_ = puntajes(modelo, cache, va[:a.n])
    out = dict(secuencias=int(min(a.n, len(va))), particion="validacion")
    for nivel, x in (("repeticion", rep_), ("serie", ser_)):
        cal = ajustar(x["y"], x["s"], x["t"])
        y, s = np.array(x["y"]), np.array(x["s"])
        pc = np.array([aplicar(cal, t, v) for t, v in zip(x["t"], s)])
        f1 = lambda p: float(2 * ((p > 0.5) & (y == 1)).sum() / max(2 * ((p > 0.5) & (y == 1)).sum() + ((p > 0.5) & (y == 0)).sum() + ((p <= 0.5) & (y == 1)).sum(), 1))
        cal.update(ejemplos=int(len(y)), positivos=int(y.sum()), auc=auc(y, s),
                   f1_sin_calibrar=f1(s), f1_calibrado=f1(pc), tipos_propios=len(cal["por_tipo"]))
        out[nivel] = cal
        print(nivel, {k: v for k, v in cal.items() if k not in ("por_tipo",)}, flush=True)
    dst = Path(a.modelo).with_name(ARCHIVO)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("guardado", dst)


if __name__ == "__main__":
    main()
