"""Video del decodificador de K3Mov: el movimiento de la persona CORREGIDO, con su mismo ritmo y cuerpo.

Paneles (Avatar 3D con la estatura del sujeto):
  1. la serie tal como se capturó (con el error de técnica), reconstruida desde las rotaciones verdaderas
  2. la versión corregida que genera el decodificador de K3Mov a partir de la captura
  3. la versión correcta verdadera del generador (misma temporización, sin el error), para comparar
Uso: python P1_K3Mov/video_correccion.py [--modelo ...] [--ejercicio sentadilla] [--error talones_elevados]
Salida: salidas/p1/correccion_<ejercicio>_<error>.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from k3mov.datos import Cache, construir_cache, armar_celdas  # noqa: E402
from nucleo import config  # noqa: E402
from nucleo.ejercicios import NOMBRES_EJ, CATALOGO_ERRORES, EJERCICIOS  # noqa: E402
from nucleo.esqueleto import Esqueleto  # noqa: E402
from nucleo.rotaciones import seis_d_a_matriz  # noqa: E402


def cuerpo_desde_rot6(rot6, raiz, estatura):
    """rot6 (T,22,6) con la articulación 0 = orientación global de la pelvis; raiz (T,3) en estaturas."""
    esq = Esqueleto(estatura)
    R = seis_d_a_matriz(np.asarray(rot6, np.float64))
    Rl = R.copy()
    Rl[:, 0] = np.eye(3)
    G, P = esq.fk(Rl, R[:, 0], np.asarray(raiz, np.float64) * estatura)
    return dict(G=G, P=P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(AQUI / "modelos" / "k3mov_v2" / "mejor.pt"))
    ap.add_argument("--ejercicio", default="sentadilla")
    ap.add_argument("--error", default=None)
    ap.add_argument("--segundos", type=float, default=8.0)
    a = ap.parse_args()
    from k3mov.modelo import K3Mov
    ck = torch.load(a.modelo, map_location="cpu", weights_only=False)
    m = K3Mov(**ck["cfg"]); m.load_state_dict(ck["estado"]); m.eval()
    c = Cache(construir_cache())
    d = c.d
    _, _, te = c.particion()
    ej_i = NOMBRES_EJ.index(a.ejercicio)
    elegido = None
    for s in te:
        if int(d["seq_ej"][s]) != ej_i:
            continue
        i0, T = int(d["seq_inicio"][s]), int(d["seq_T"][s])
        err = np.asarray(d["err"][i0:i0 + T], np.float32)
        nombres = [CATALOGO_ERRORES[g].split(":")[1] if g >= 0 else None for g in d["seq_err_idx"][s]]
        activos = [n for j, n in enumerate(nombres) if n and err[:, j].max() > 0.5]
        if (a.error and a.error in activos) or (not a.error and activos):
            elegido = (s, activos[0] if not a.error else a.error)
            break
    if elegido is None:
        raise SystemExit("no hay una serie de prueba con ese error")
    s, nombre_err = elegido
    i0, T = int(d["seq_inicio"][s]), int(d["seq_T"][s])
    rep = np.asarray(d["rep"][i0:i0 + T])
    a0 = int(np.flatnonzero(rep > 0)[0]) if (rep > 0).any() else 0
    b0 = min(T, a0 + int(a.segundos * 30))
    sl = slice(i0 + a0, i0 + b0)
    suj = np.asarray(d["seq_sujeto"][s], np.float64)
    est = float(suj[0])
    cel, P = armar_celdas(A=np.asarray(d["angulos_captura"][sl], np.float64), L=np.asarray(d["captura_T"][sl], np.float64),
                          vis=np.asarray(d["captura_vis"][sl], np.float64), imu=np.asarray(d["imu"][sl], np.float64),
                          sujeto=suj, T=b0 - a0)
    with torch.no_grad():
        o = m({k: torch.from_numpy(v[None].astype(np.float32)) for k, v in cel.items()}, torch.from_numpy(P[None]),
              causal=True, decodificar=True)
    corregido = cuerpo_desde_rot6(o["dec"]["rot6"][0].numpy(), o["dec"]["raiz"][0].numpy(), est)
    # verdad: la serie con el error (captura) y su versión correcta, desde la caché
    from k3mov.datos import _rot_objetivo
    raiz_c = np.asarray(d["raiz_t_corr"][sl], np.float64)
    correcta = cuerpo_desde_rot6(_rot_objetivo(d["R6_corr"][sl], d["raiz6_corr"][sl]), (raiz_c - raiz_c[0]) / est, est)
    # la serie con el error: su cinemática exacta (G6, P) está en el fragmento del conjunto, por su semilla
    import glob
    sem = int(d["seq_semilla"][s])
    usuario = None
    for f in sorted(glob.glob(str(config.DATOS_GEN / config.CONJUNTO / "frag_*.npz"))):
        zf = np.load(f)
        for i in sorted({int(k.split("__")[0]) for k in zf.files}):
            if int(zf[f"{i}__semilla"]) == sem:
                G6 = zf[f"{i}__G6"].astype(np.float64)[a0:b0]
                Pf = zf[f"{i}__P"].astype(np.float64)[a0:b0]
                usuario = dict(G=seis_d_a_matriz(G6), P=Pf)
                break
        if usuario is not None:
            break
    from nucleo.avatar_malla import AvatarMalla
    from nucleo.render import video_lado_a_lado, CIAN, AMBAR
    from P5_AtlasAvatar3D.atlas.personalizar import a_malla
    malla = AvatarMalla(estatura=est)
    Tm = min(len(usuario["G"]), len(corregido["G"]), len(correcta["G"]))
    cuerpos = [dict(tipo="malla", V=a_malla(malla, x["G"][:Tm], x["P"][:Tm]), F=malla.F, color=col)
               for x, col in ((usuario, AMBAR), (corregido, CIAN), (correcta, (120, 200, 120)))]
    sal = config.RAIZ / "salidas" / "p1"
    sal.mkdir(parents=True, exist_ok=True)
    ruta = sal / f"correccion_{a.ejercicio}_{nombre_err}.mp4"
    video_lado_a_lado(cuerpos, ruta, titulos=[f"Captura ({nombre_err.replace('_', ' ')})", "Corregido por K3Mov",
                                              "Correcto (verdad)"], pie="ElectroBiomed · Avatar 3D", ancho=400, alto=560)
    print("video:", ruta, " fotogramas", Tm, " ejercicio", EJERCICIOS[a.ejercicio]["nombre"])


if __name__ == "__main__":
    main()
