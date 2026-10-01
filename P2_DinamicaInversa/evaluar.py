"""Evaluación comparada del Proyecto 2 sobre secuencias de prueba (sujetos no vistos):

  A · Física directa sobre la captura ruidosa: reajuste del esqueleto a las marcas «world»,
      traslación por la imagen, Butterworth de 6 Hz y Newton–Euler con reparto de reacciones.
  B · TCN caja negra.
  C · Híbrido: TCN estima el estado cinemático limpio -> Newton–Euler exacto.
Métricas: RMSE de momentos (N·m/kg) por grupo articular, de la reacción vertical de cada pie (% del
peso) y de la potencia (W/kg); coste (parámetros, milisegundos por fotograma).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from dinv.datos import abrir_cache, entradas  # noqa: E402
from dinv.modelos import TCN  # noqa: E402
from entrenar import particion  # noqa: E402
from nucleo.esqueleto import Esqueleto, PADRES, NJ_CUERPO, MP  # noqa: E402
from nucleo.antropometria import ModeloInercial  # noqa: E402
from nucleo.ejercicios import NOMBRES_EJ, EJERCICIOS  # noqa: E402
from nucleo.retarget import reajustar  # noqa: E402
from nucleo.rotaciones import seis_d_a_matriz  # noqa: E402
from nucleo import dinamica as D  # noqa: E402

CLAVES_M = ['caderaD_abd', 'caderaD_ext', 'caderaD_rotint', 'caderaI_abd', 'caderaI_ext', 'caderaI_rotint', 'codoD_flex',
            'codoI_flex', 'cuello_ext', 'hombroD_abd', 'hombroD_flex', 'hombroD_rotint', 'hombroI_abd', 'hombroI_flex',
            'hombroI_rotint', 'lumbar_ext', 'lumbar_lat', 'lumbar_rot', 'munecaD_flex', 'munecaI_flex', 'rodillaD_ext',
            'rodillaI_ext', 'tobilloD_plantar', 'tobilloI_plantar']
GRUPOS = {"cadera": ["caderaD_ext", "caderaI_ext", "caderaD_abd", "caderaI_abd"], "rodilla": ["rodillaD_ext", "rodillaI_ext"],
          "tobillo": ["tobilloD_plantar", "tobilloI_plantar"], "lumbar": ["lumbar_ext"],
          "hombro": ["hombroD_flex", "hombroI_flex", "hombroD_abd", "hombroI_abd"], "codo": ["codoD_flex", "codoI_flex"]}


def cargar(ruta):
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = TCN(c["f_in"], c["salidas"], c["c"], c["niveles"], c["causal"])
    m.load_state_dict(ck["estado"])
    m.eval()
    return m


def modelo_inercial(esq, suj, carga):
    mod = ModeloInercial(esq, float(suj[1]), "H" if suj[2] > 0.5 else "M")
    if carga > 0:
        for j in (20, 21):
            m0 = mod.m[j]
            cm = mod.c[j] + np.array([0, -0.08 * esq.s, 0])
            mod.c[j] = (mod.c[j] * m0 + cm * carga) / (m0 + carga)
            mod.m[j] = m0 + carga
    return mod


def fisica_sobre(G, P, suj, carga, ej, fps=30.0):
    esq = Esqueleto(float(suj[0]))
    mod = modelo_inercial(esq, suj, carga)
    spec = EJERCICIOS[ej]
    grupos = spec["apoyos"]
    h_extra = {"gluteos": 0.10, "espalda": 0.08} if spec.get("supino") else {}
    if spec.get("colgado"):
        L = esq.marcas(G, P)
        suelo = float(np.median(L[:, [MP["MUNECA_I"], MP["MUNECA_D"]], 1]))
        out = D.dinamica_inversa(esq, mod, G, P, fps, grupos=grupos, suelo=suelo,
                                 contactos={g: np.ones(len(G), bool) for g in grupos})
    else:
        L = esq.marcas(G, P)
        suelo = float(np.percentile(L[:, [29, 30, 31, 32], 1].min(1), 5))
        out = D.dinamica_inversa(esq, mod, G, P, fps, grupos=grupos, suelo=suelo, h_extra=h_extra,
                                 contactos=({g: L[:, D.GRUPOS_CONTACTO[g], 1].min(1) < suelo + 0.015 for g in grupos}
                                            if spec.get("libre") else None))
    M = np.stack([out["anat"][k] for k in CLAVES_M], -1)
    F = np.full((len(G), 2, 3), np.nan)
    for k, g in enumerate(("pie_D", "pie_I")):
        if g in out["reacciones"]:
            F[:, k] = out["reacciones"][g]["F"]
    return M, F, out["potencia"]


def estado_a_GP(G6, raiz, suj):
    esq = Esqueleto(float(suj[0]))
    G = seis_d_a_matriz(G6.reshape(len(G6), 22, 6).astype(np.float64))
    T = len(G)
    P = np.zeros((T, 22, 3))
    P[:, 0] = raiz * float(suj[0])
    for j in range(1, NJ_CUERPO):
        p = PADRES[j]
        P[:, j] = P[:, p] + np.einsum("tab,b->ta", G[:, p], esq.offsets[j])
    return G, P


def metricas(Mp, Mt, Fp, Ft, Pp, Pt, masa):
    out = {}
    for g, ks in GRUPOS.items():
        idx = [CLAVES_M.index(k) for k in ks]
        e = (Mp[:, idx] - Mt[:, idx]) / masa
        out[f"rmse_{g}_Nm_kg"] = float(np.sqrt(np.nanmean(e ** 2)))
    e = (Mp - Mt) / masa
    out["rmse_momentos_Nm_kg"] = float(np.sqrt(np.nanmean(e ** 2)))
    ok = np.isfinite(Ft[..., 1]) & np.isfinite(Fp[..., 1])
    if ok.any():
        out["rmse_reaccion_vertical_pctBW"] = float(np.sqrt(np.mean(((Fp[..., 1] - Ft[..., 1])[ok] / (masa * 9.81)) ** 2)) * 100)
    out["rmse_potencia_W_kg"] = float(np.sqrt(np.nanmean(((Pp - Pt) / masa) ** 2)))
    return out


def main():
    torch.set_num_threads(6)
    z = abrir_cache()
    _, _, te = particion(len(z["seq_T"]))
    te = te[:int(sys.argv[1]) if len(sys.argv) > 1 else 60]
    tcn = cargar(AQUI / "modelos" / "tcn_c96" / "mejor.pt")
    est = cargar(AQUI / "modelos" / "estado_c96" / "mejor.pt")
    ruta_vivo = AQUI / "modelos" / "tcn_vivo_c96" / "mejor.pt"
    vivo = cargar(ruta_vivo) if ruta_vivo.exists() else None
    res = {"A_fisica_captura": [], "B_tcn": [], "C_hibrido": [], "A_fisica_limpia": []}
    if vivo is not None:
        res["B_tcn_vivo"] = []
    tiempos = {k: [] for k in res}
    for n, s in enumerate(te):
        i0, T = int(z["seq_inicio"][s]), int(z["seq_T"][s])
        sl = slice(i0, i0 + T)
        suj, carga = z["seq_sujeto"][s], float(z["seq_carga"][s])
        ej = NOMBRES_EJ[int(z["seq_ej"][s])]
        masa = float(suj[1])
        Mt = z["momentos"][sl].astype(np.float64)
        Ft = z["grf"][sl].astype(np.float64)
        Pt = z["potencia"][sl].astype(np.float64)
        X = torch.from_numpy(entradas(z["mundo_captura"][sl].astype(np.float64), z["captura_T"][sl].astype(np.float64), suj, carga))[None]
        # B
        t0 = time.time()
        with torch.no_grad():
            o = tcn(X)
        tiempos["B_tcn"].append((time.time() - t0) / T * 1000)
        res["B_tcn"].append(metricas(o["M"][0].numpy() * masa, Mt, o["F"][0].numpy().reshape(T, 2, 3) * masa * 9.81, Ft,
                                     o["Pw"][0].numpy() * masa, Pt, masa))
        if vivo is not None:                                           # B causal (solo pasado: apta para vivo)
            t0 = time.time()
            with torch.no_grad():
                o = vivo(X)
            tiempos["B_tcn_vivo"].append((time.time() - t0) / T * 1000)
            res["B_tcn_vivo"].append(metricas(o["M"][0].numpy() * masa, Mt, o["F"][0].numpy().reshape(T, 2, 3) * masa * 9.81,
                                              Ft, o["Pw"][0].numpy() * masa, Pt, masa))
        # C
        t0 = time.time()
        with torch.no_grad():
            o = est(X)
        G, P = estado_a_GP(o["G6"][0].numpy(), o["raiz"][0].numpy(), suj)
        P = P + z["P"][i0, 0].astype(np.float64)                       # misma referencia inicial que la verdad
        Mp, Fp, Pp = fisica_sobre(G, P, suj, carga, ej)
        tiempos["C_hibrido"].append((time.time() - t0) / T * 1000)
        res["C_hibrido"].append(metricas(Mp, Mt, Fp, Ft, Pp, Pt, masa))
        # A (física directa sobre la captura: reajuste + traslación por la imagen)
        t0 = time.time()
        W = z["mundo_captura"][sl].astype(np.float64)
        r = reajustar(W, float(suj[0]), Esqueleto(float(suj[0])))
        capT = z["captura_T"][sl].astype(np.float64)
        mh = capT[:, [23, 24]].mean(1)
        tor = np.median(np.linalg.norm(capT[:, [11, 12]].mean(1) - mh, axis=-1))
        esc = 0.288 * float(suj[0]) / max(tor, 1e-6)                    # tronco real / tronco en la imagen
        tras = (mh - mh[0]) * esc
        tras[:, 2] = 0.0                                                # la profundidad no se observa
        if len(tras) > 12:                                              # el ruido de la imagen, derivado dos
            b_, a_ = butter(2, 2.0 / 15.0)                              # veces, daría fuerzas falsas: 2 Hz
            tras = filtfilt(b_, a_, tras, axis=0)
        # traslación SOLO desde la imagen: la del reajuste sale del centro de cadera ruidoso de la captura
        P_A = r["P"] - r["P"][:, :1] + z["P"][i0, 0].astype(np.float64) + tras[:, None, :]
        Mp, Fp, Pp = fisica_sobre(r["G"], P_A, suj, carga, ej)
        tiempos["A_fisica_captura"].append((time.time() - t0) / T * 1000)
        res["A_fisica_captura"].append(metricas(Mp, Mt, Fp, Ft, Pp, Pt, masa))
        # techo: física sobre la cinemática limpia (debe reproducir la verdad salvo el filtrado)
        Gc = seis_d_a_matriz(z["G6"][sl].astype(np.float64))
        Mp, Fp, Pp = fisica_sobre(Gc, z["P"][sl].astype(np.float64), suj, carga, ej)
        res["A_fisica_limpia"].append(metricas(Mp, Mt, Fp, Ft, Pp, Pt, masa))
        if n % 10 == 0:
            print(n, ej, {k: round(v[-1]["rmse_momentos_Nm_kg"], 3) for k, v in res.items()}, flush=True)
    resumen = {}
    for k, v in res.items():
        claves = v[0].keys()
        resumen[k] = {c: float(np.nanmean([x.get(c, np.nan) for x in v])) for c in claves}
        if tiempos[k]:
            resumen[k]["ms_por_fotograma"] = float(np.mean(tiempos[k]))
    resumen["parametros"] = dict(B_tcn=sum(p.numel() for p in tcn.parameters()), C_hibrido=sum(p.numel() for p in est.parameters()))
    if vivo is not None:
        resumen["parametros"]["B_tcn_vivo"] = sum(p.numel() for p in vivo.parameters())
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "comparacion.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
