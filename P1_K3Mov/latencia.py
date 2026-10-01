"""Latencia de K3Mov en la CPU: en vivo (un fotograma por llamada, con memoria) y por serie completa.

Conviene correrlo con la máquina libre: con otros cómputos en marcha la cifra sale inflada.
Uso:  python P1_K3Mov/latencia.py [--modelo ...] [--series 6] [--hilos 1]
Salida: <carpeta del modelo>/latencia.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from k3mov.datos import Cache, construir_cache, armar_celdas  # noqa: E402
from k3mov.en_vivo import K3MovEnVivo  # noqa: E402
from k3mov.modelo import K3Mov, NOMBRES_CELDAS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(AQUI / "modelos" / "k3mov_v2" / "mejor.pt"))
    ap.add_argument("--series", type=int, default=6)
    ap.add_argument("--hilos", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    ck = torch.load(a.modelo, map_location="cpu", weights_only=False)
    m = K3Mov(**ck["cfg"]); m.load_state_dict(ck["estado"]); m.eval()
    c = Cache(construir_cache())
    d = c.d
    _, _, te = c.particion()
    vivo = {True: [], False: []}
    bloques, serie = [], []
    for s in te[:a.series]:
        i0, T = int(d["seq_inicio"][s]), int(d["seq_T"][s])
        sl = slice(i0, i0 + T)
        cel, P = armar_celdas(A=d["angulos_captura"][sl].astype(np.float64), L=d["captura_T"][sl].astype(np.float64),
                              vis=d["captura_vis"][sl].astype(np.float64), imu=d["imu"][sl].astype(np.float64),
                              sujeto=d["seq_sujeto"][s], T=T)
        for dec in (False, True):
            ev = K3MovEnVivo(m)
            for t in range(T):
                ev.paso({k: cel[k][t] for k in NOMBRES_CELDAS}, P[t], decodificar=dec)
            vivo[dec].extend(ev.lat_ms[10:])              # sin los primeros (calentamiento)
        ev = K3MovEnVivo(m)                               # bloques de 5 fotogramas, como la aplicación
        for a0 in range(0, T, 5):
            ev.bloque({k: cel[k][a0:a0 + 5] for k in NOMBRES_CELDAS}, P[a0:a0 + 5], decodificar=False)
        bloques.extend(ev.lat_ms[2:])
        with torch.no_grad():
            t0 = time.perf_counter()
            m({k: torch.from_numpy(v[None].astype(np.float32)) for k, v in cel.items()}, torch.from_numpy(P[None]),
              causal=True, decodificar=True)
            serie.append((time.perf_counter() - t0) * 1000 / T)
    pc = lambda x: [round(float(v), 3) for v in np.percentile(x, [50, 95])]
    out = dict(hilos=a.hilos, series=a.series,
               en_vivo_ms_p50_p95=pc(vivo[False]), en_vivo_con_postura_correcta_ms_p50_p95=pc(vivo[True]),
               en_vivo_bloques_de_5_ms_por_fotograma_p50=round(float(np.median(bloques)), 3),
               serie_completa_ms_por_fotograma=round(float(np.mean(serie)), 3),
               fotogramas_medidos=len(vivo[False]))
    ruta = Path(a.modelo).parent / "latencia.json"
    ruta.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
