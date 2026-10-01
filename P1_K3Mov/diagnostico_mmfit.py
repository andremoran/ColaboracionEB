"""Diferencias entre las entradas reales de MM-Fit y las del conjunto sintético, por ejercicio (sin modelo).

Mide lo que explica la brecha entre lo sintético y lo real: la postura (sentado o de pie: flexión de cadera y
rodilla), la inclinación del tronco en ejercicios que se hacen erguidos (delata la inclinación de la cámara),
si los lados se alternan (curl con un brazo cada vez, zancada que cambia de pierna) y qué ángulos faltan porque
el esqueleto de MM-Fit (17 puntos) no tiene pies, manos ni cara.
Uso:  python P1_K3Mov/diagnostico_mmfit.py
Salida: P1_K3Mov/resultados/diagnostico_mmfit.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from k3mov.datos import Cache, construir_cache  # noqa: E402
from nucleo import bases_reales as BR  # noqa: E402
from nucleo.angulos_clinicos import CLAVES  # noqa: E402
from nucleo.ejercicios import NOMBRES_EJ  # noqa: E402

MEDIDAS = ("hipR_flex", "kneeR_flex", "trunk_flex")
PAR = {"curl_biceps": ("elbowR_flex", "elbowL_flex"), "press_hombro": ("elbowR_flex", "elbowL_flex"),
       "sentadilla": ("kneeR_flex", "kneeL_flex"), "remo": ("elbowR_flex", "elbowL_flex")}


def correlacion(A, par):
    x, y = A[:, CLAVES.index(par[0])], A[:, CLAVES.index(par[1])]
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else float("nan")


def pierna_adelantada(A):
    """En cada fondo de la zancada, qué cadera está más flexionada (la de la pierna adelantada)."""
    k = np.nan_to_num(A[:, CLAVES.index("kneeR_flex")] + A[:, CLAVES.index("kneeL_flex")])
    picos, _ = find_peaks(k, distance=30, prominence=20)
    return ["D" if A[p, CLAVES.index("hipR_flex")] > A[p, CLAVES.index("hipL_flex")] else "I" for p in picos]


def mediana(x):
    return round(float(np.nanmedian(x)), 1)


def main():
    real = {}
    for w in BR.MMFIT_TEST + BR.MMFIT_TEST_NUEVOS:
        try:
            s = BR.mmfit_sesion(w, con_imu=False)
        except FileNotFoundError:
            continue
        for (a, b, n, nombre) in s["series"]:
            mk = (s["frames"] >= a) & (s["frames"] <= b)
            real.setdefault(nombre, []).append(s["angulos"][mk])
    c = Cache(construir_cache())
    d = c.d
    _, _, te = c.particion()
    sint = {}
    for s_ in te:
        e = NOMBRES_EJ[int(d["seq_ej"][s_])]
        if e in real and len(sint.setdefault(e, [])) < 20:
            i0, T = int(d["seq_inicio"][s_]), int(d["seq_T"][s_])
            sint[e].append(np.asarray(d["angulos_captura"][i0:i0 + T], np.float64))
    out = {}
    cambios = []
    for A in real.get("zancada", []):
        lados = pierna_adelantada(A)
        cambios += [int(a != b) for a, b in zip(lados, lados[1:])]
    for e, lista in real.items():
        A = np.concatenate(lista)
        S = np.concatenate(sint[e]) if sint.get(e) else None
        out[e] = dict(series=len(lista),
                      real={k: mediana(A[:, CLAVES.index(k)]) for k in MEDIDAS},
                      sintetico={k: mediana(S[:, CLAVES.index(k)]) for k in MEDIDAS} if S is not None else None,
                      angulos_ausentes=[k for j, k in enumerate(CLAVES) if np.isnan(A[:, j]).mean() > 0.5])
        if e in PAR:
            out[e]["correlacion_der_izq"] = dict(real=round(float(np.nanmedian([correlacion(x, PAR[e]) for x in lista])), 2),
                                                 sintetico=round(float(np.nanmedian([correlacion(x, PAR[e]) for x in sint.get(e, [])])), 2)
                                                 if sint.get(e) else None)
    if cambios:
        out["zancada"]["cambia_de_pierna_pct"] = round(100 * float(np.mean(cambios)), 1)
    dst = AQUI / "resultados" / "diagnostico_mmfit.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for e, r in out.items():
        print(f"{e:18s} real {r['real']}  sintético {r['sintetico']}  ausentes {len(r['angulos_ausentes'])}")


if __name__ == "__main__":
    main()
