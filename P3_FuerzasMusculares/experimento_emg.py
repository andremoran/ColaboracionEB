"""¿Cuánto mejoran dos canales de EMG la estimación de fuerzas musculares? Calibración genética.

Situación real de la plataforma: dos sensores de superficie (uMyo) sobre el vasto lateral y el
gastrocnemio medial derechos. La optimización estática de mínimo esfuerzo supone una estrategia
«económica»; una persona real co-contrae (isquiotibiales y gemelos trabajan más de lo mínimo).

Verdad simulada: optimización estática con otra estrategia (isquiotibiales y gemelos «más baratos»,
es decir, co-contracción). EMG medida: la activación verdadera del músculo, retrasada por el retardo
electromecánico, dividida por una ganancia desconocida y con ruido multiplicativo.

Estimadores comparados sobre fotogramas de prueba:
  1. optimización estática sola (mínimo esfuerzo);
  2. guiada por EMG con la normalización ingenua (EMG / su máximo);
  3. guiada por EMG con ganancia y retardo calibrados por un ALGORITMO GENÉTICO: cada individuo es
     (ganancia, retardo) por canal; su aptitud es el esfuerzo + reservas de la solución guiada en
     fotogramas de calibración (la ganancia verdadera es la que hace el EMG compatible con un reparto
     de mínimo esfuerzo del resto de músculos);
  4. guiada por EMG con la ganancia y el retardo verdaderos (techo).
Salida: resultados/experimento_emg.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from musculos.myoleg import Musculos  # noqa: E402
from musculos.optimizacion import resolver  # noqa: E402
from nucleo import config  # noqa: E402
from nucleo.ejercicios import NOMBRES_EJ  # noqa: E402

CANALES = ["vaslat_r", "gasmed_r"]
COCONTRAE = ("bflh_r", "bfsh_r", "semimem_r", "semiten_r", "gasmed_r", "gaslat_r")


def _desplazar(x, d):
    """x retrasado d muestras (d >= 0) con el primer valor repetido."""
    d = int(round(d))
    return x if d <= 0 else np.r_[np.full(d, x[0]), x[:-d]]


def main(n_seq=6, ejercicios=("sentadilla", "zancada", "peso_muerto"), poblacion=14, generaciones=10, semilla=0):
    rng = np.random.default_rng(semilla)
    z = np.load(config.DATOS_GEN / config.SO_MYOLEG)
    nombres = Musculos().nombres
    idx_can = [nombres.index(c) for c in CANALES]
    esfuerzo_real = np.ones(len(nombres))
    for c in COCONTRAE:
        esfuerzo_real[nombres.index(c)] = 0.35                    # la persona recurre más a estos músculos
    geo = z["geo_pos"]
    off = z["seq_off"]
    ej_idx = [NOMBRES_EJ.index(e) for e in ejercicios]
    seqs = [s for s in range(len(off)) if int(z["seq_ej"][s]) in ej_idx]
    seqs = list(rng.choice(seqs, size=min(n_seq, len(seqs)), replace=False))
    res = {k: [] for k in ("so", "emg_ingenua", "emg_genetica", "emg_verdadera")}
    res_nomed = {k: [] for k in res}
    ganancias = []
    t0 = time.time()
    for s in seqs:
        o, l = off[s]
        g_idx = np.flatnonzero((geo >= o) & (geo < o + l))       # fotogramas con geometría guardada
        if len(g_idx) < 16:
            continue
        R = z["R"][g_idx].astype(np.float64)
        Fa, Fp = z["Fa"][g_idx].astype(np.float64), z["Fp"][g_idx].astype(np.float64)
        tau = z["tau"][geo[g_idx]].astype(np.float64)
        # verdad: estrategia con co-contracción
        a_real, _, _ = resolver(R, Fa, Fp, tau, esfuerzo=esfuerzo_real)
        # EMG medida: retardo 1-2 muestras (a 10 Hz: 100-200 ms incluido el filtrado), ganancia desconocida, ruido
        g_true = rng.uniform(0.6, 1.6, len(CANALES))
        d_true = rng.integers(1, 3, len(CANALES))
        emg = {}
        for c, i in enumerate(idx_can):
            x = a_real[:, i]
            medida = np.r_[x[d_true[c]:], np.full(d_true[c], x[-1])] / g_true[c]      # la EMG adelanta a la fuerza
            emg[i] = np.clip(medida * (1 + 0.08 * rng.normal(size=len(x))), 0, None)
        # partición calibración / prueba (mitades temporales)
        n = len(g_idx)
        cal, pru = np.arange(n // 2), np.arange(n // 2, n)

        def guiada(g, d, filas):
            e = {i: np.clip(g[c] * _desplazar(emg[i], d[c])[filas], 0, 1) for c, i in enumerate(idx_can)}
            return resolver(R[filas], Fa[filas], Fp[filas], tau[filas], emg=e)

        def aptitud(ind):
            g, d = ind[:len(CANALES)], ind[len(CANALES):]
            a, _, rsv = guiada(g, d, cal)
            return float((a ** 2).sum(1).mean() + 0.02 * (rsv ** 2).sum(1).mean())

        # algoritmo genético: selección por torneo, cruce aritmético, mutación gaussiana, elitismo
        pob = np.c_[rng.uniform(0.3, 2.5, (poblacion, len(CANALES))), rng.integers(0, 4, (poblacion, len(CANALES)))]
        apt = np.array([aptitud(p) for p in pob])
        for _ in range(generaciones):
            nueva = [pob[np.argmin(apt)].copy()]
            while len(nueva) < poblacion:
                a1, a2 = [pob[min(rng.choice(poblacion, 2, replace=False), key=lambda i: apt[i])] for _ in range(2)]
                w = rng.random()
                h = w * a1 + (1 - w) * a2
                h[:len(CANALES)] *= np.exp(0.15 * rng.normal(size=len(CANALES)))
                h[len(CANALES):] = np.clip(np.round(h[len(CANALES):] + rng.integers(-1, 2, len(CANALES)) * (rng.random(len(CANALES)) < 0.3)), 0, 4)
                nueva.append(h)
            pob = np.array(nueva)
            apt = np.array([aptitud(p) for p in pob])
        mejor = pob[np.argmin(apt)]
        g_ga, d_ga = mejor[:len(CANALES)], mejor[len(CANALES):]
        ganancias.append(dict(verdadera=g_true.tolist(), genetica=g_ga.tolist(), retardo_verdadero=d_true.tolist(),
                              retardo_genetico=d_ga.tolist()))
        # evaluación en la mitad de prueba
        a_so, _, _ = resolver(R[pru], Fa[pru], Fp[pru], tau[pru])
        g_ing = np.array([1.0 / max(emg[i].max(), 1e-6) for i in idx_can])
        a_ing, _, _ = guiada(g_ing, np.zeros(len(CANALES)), pru)
        a_ga, _, _ = guiada(g_ga, d_ga, pru)
        a_ver, _, _ = guiada(g_true, d_true.astype(float), pru)       # medida = a/g  ->  a = g·medida
        no_medidos = np.setdiff1d(np.arange(len(nombres)), idx_can)
        for k, a in (("so", a_so), ("emg_ingenua", a_ing), ("emg_genetica", a_ga), ("emg_verdadera", a_ver)):
            res[k].append(float(np.abs(a - a_real[pru]).mean()))
            res_nomed[k].append(float(np.abs(a[:, no_medidos] - a_real[pru][:, no_medidos]).mean()))
        print(f"serie {s} ({NOMBRES_EJ[int(z['seq_ej'][s])]}): MAE activación "
              + ", ".join(f"{k} {res[k][-1]:.4f}" for k in res) + f"  ({time.time() - t0:.0f}s)", flush=True)
    out = dict(canales=CANALES, cocontraccion=list(COCONTRAE), series=len(res["so"]),
               mae_activacion={k: float(np.mean(v)) for k, v in res.items()},
               mae_activacion_no_medidos={k: float(np.mean(v)) for k, v in res_nomed.items()},
               ganancias=ganancias, segundos=round(time.time() - t0))
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "experimento_emg.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("mae_activacion", "mae_activacion_no_medidos")}, indent=1))


if __name__ == "__main__":
    main()
