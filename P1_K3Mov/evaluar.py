"""Evaluación de K3Mov: prueba sintética (sujetos no vistos), MM-Fit real (sujetos de prueba) y
EC3D real (errores de técnica etiquetados).

Uso: python P1_K3Mov/evaluar.py --modelo P1_K3Mov/modelos/k3mov_d64/mejor.pt [--mmfit] [--ec3d]
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

from k3mov.datos import Cache, construir_cache, armar_celdas, N_ERR, NINGUNO  # noqa: E402
from k3mov.modelo import K3Mov, NOMBRES_CELDAS  # noqa: E402
from k3mov.decodificacion import decodificar_serie, viterbi, matriz_transicion, filtro_en_vivo  # noqa: E402
from nucleo.ejercicios import NOMBRES_EJ, CATALOGO_ERRORES, EJERCICIOS  # noqa: E402
from nucleo import bases_reales as BR  # noqa: E402
from nucleo.fk_torch import FKTorch  # noqa: E402
from k3mov import calibracion  # noqa: E402


def cargar(ruta):
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    m = K3Mov(**ck["cfg"])
    m.load_state_dict(ck["estado"])
    m.eval()
    return m


@torch.no_grad()
def inferir(modelo, cel, P, decodificar=True):
    c = {k: torch.from_numpy(v)[None] for k, v in cel.items()}
    out = modelo(c, torch.from_numpy(P)[None], causal=True, decodificar=decodificar)
    r = dict(p_ej=torch.softmax(out["ejercicio"][0], -1).numpy(), p_fase=torch.softmax(out["fase"][0], -1).numpy(),
             dens=out["rep"][0].numpy(), p_err=torch.sigmoid(out["errores"][0]).numpy(),
             calidad=out["calidad"][0].numpy(), fatiga=out["fatiga"][0].numpy())
    if decodificar:
        r["rot6"] = out["dec"]["rot6"][0]
        r["ang_dec"] = out["dec"]["angulos"][0].numpy() * 90
    return r


def f1_segmental(pred, real, clases=(1, 3), iou=0.5):
    """F1@IoU sobre segmentos de fases de movimiento (métrica usual de segmentación de acciones)."""
    def segs(x):
        out, i = [], 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[j + 1] == x[i]:
                j += 1
            if x[i] in clases:
                out.append((x[i], i, j))
            i = j + 1
        return out
    P, R = segs(pred), segs(real)
    usados = set()
    tp = 0
    for c, a, b in P:
        mejor, k = 0, None
        for n, (c2, a2, b2) in enumerate(R):
            if c2 != c or n in usados:
                continue
            inter = max(0, min(b, b2) - max(a, a2) + 1)
            uni = max(b, b2) - min(a, a2) + 1
            if inter / uni > mejor:
                mejor, k = inter / uni, n
        if mejor >= iou:
            tp += 1
            usados.add(k)
    prec = tp / max(len(P), 1)
    rec = tp / max(len(R), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


# ================================================================ sintético ===
def evaluar_sintetico(modelo, cache, idx, n_max=None, calib=None):
    d = cache.d
    fk = FKTorch()
    res = dict(ej_seq=[], ej_frame=[], fase_arg=[], fase_vit=[], fase_viv=[], f1_vit=[], conteo_abs=[], conteo_obo=[],
               conteo_exacto=[], t_con_err=[], t_exc_err=[], cal_mae=[], fat_mae=[], mpjpe_mm=[], ang_mae=[],
               err_y=[], err_s=[], err_tipo=[], lat_ms=[])
    for n, s in enumerate(idx[:n_max] if n_max else idx):
        i0, T = int(d["seq_inicio"][s]), int(d["seq_T"][s])
        sl = slice(i0, i0 + T)
        cel, P = armar_celdas(A=d["angulos_captura"][sl].astype(np.float64), L=d["captura_T"][sl].astype(np.float64),
                              vis=d["captura_vis"][sl].astype(np.float64), imu=d["imu"][sl].astype(np.float64),
                              act=None, sujeto=d["seq_sujeto"][s], T=T)
        t0 = time.time()
        r = inferir(modelo, cel, P)
        res["lat_ms"].append((time.time() - t0) * 1000 / T)
        ej_true = int(d["seq_ej"][s])
        fase = d["fase"][sl].astype(int)
        act = fase > 0
        res["ej_seq"].append(int(r["p_ej"][act, :NINGUNO].sum(0).argmax() == ej_true))
        y_ej = np.where(act, ej_true, NINGUNO)
        res["ej_frame"].append(float((r["p_ej"].argmax(-1) == y_ej).mean()))
        prim = EJERCICIOS[NOMBRES_EJ[ej_true]]["primera"]
        Tm = matriz_transicion(prim)
        lp = np.log(np.clip(r["p_fase"], 1e-6, 1))
        fv = viterbi(lp, Tm)
        fl = filtro_en_vivo(lp, Tm)
        res["fase_arg"].append(float((r["p_fase"].argmax(-1) == fase).mean()))
        res["fase_vit"].append(float((fv == fase).mean()))
        res["fase_viv"].append(float((fl == fase).mean()))
        res["f1_vit"].append(f1_segmental(fv, fase))
        ev = cache.ev_por_seq[s]
        n_true = len(ev)
        cont = float(r["dens"].sum())
        if prim != "iso":
            res["conteo_abs"].append(abs(cont - n_true))
            res["conteo_obo"].append(int(abs(round(cont) - n_true) <= 1))
            res["conteo_exacto"].append(int(round(cont) == n_true))
            # tempos por repetición: segmentos de la fase decodificada dentro de cada repetición real
            for e in ev:
                a, b = int(round(e[2] * 30)), int(round(e[7] * 30))
                seg = fv[a:b + 1]
                res["t_con_err"].append(abs((seg == 1).sum() / 30.0 - e[8]))
                res["t_exc_err"].append(abs((seg == 3).sum() / 30.0 - e[9]))
        err_true = d["err"][sl].astype(np.float32)
        ei = d["seq_err_idx"][s]
        rep = d["rep"][sl].astype(int)
        for j, g in enumerate(ei):
            if g < 0:
                continue
            for k in np.unique(rep[rep > 0]):
                m = rep == k
                res["err_y"].append(int(err_true[m, j].max() > 0.3))
                res["err_s"].append(float(r["p_err"][m, g].mean()))
                res["err_tipo"].append(CATALOGO_ERRORES[g])
        cal = 1 - err_true.max(-1)
        if act.any():
            res["cal_mae"].append(float(np.abs(r["calidad"][act] - cal[act]).mean()))
        res["fat_mae"].append(float(np.abs(r["fatiga"] - d["fatiga"][sl].astype(np.float32)).mean()))
        # decodificador: marcas por cinemática directa frente al movimiento correcto
        est = torch.tensor([float(d["seq_sujeto"][s][0])])
        marcas = fk(r["rot6"][None], est)[0].numpy()
        Lc = d["L_corr"][sl].astype(np.float32)
        a1 = marcas - marcas[:, [23, 24]].mean(1, keepdims=True)
        a2 = Lc - Lc[:, [23, 24]].mean(1, keepdims=True)
        res["mpjpe_mm"].append(float(np.linalg.norm(a1 - a2, axis=-1).mean() * 1000))
        Ac = d["angulos_corr"][sl].astype(np.float32)
        ok = np.isfinite(Ac)
        res["ang_mae"].append(float(np.abs(r["ang_dec"][ok] - Ac[ok]).mean()))
    y, sc = np.array(res["err_y"]), np.array(res["err_s"])
    tipos = sorted(set(res["err_tipo"]))
    aucs = {t: auc(y[np.array(res["err_tipo"]) == t], sc[np.array(res["err_tipo"]) == t]) for t in tipos}
    f1 = lambda pred: float(2 * (pred & (y == 1)).sum() / max(2 * (pred & (y == 1)).sum() + (pred & (y == 0)).sum()
                                                           + (~pred & (y == 1)).sum(), 1))
    f1_crudo = f1(sc > 0.5)
    if calib is not None:            # probabilidades calibradas con la partición de validación
        sc = np.array([calibracion.aplicar(calib["repeticion"], t, v) for t, v in zip(res["err_tipo"], sc)])
    resumen = dict(
        secuencias=len(res["ej_seq"]), acc_ejercicio_serie=np.mean(res["ej_seq"]), acc_ejercicio_fotograma=np.mean(res["ej_frame"]),
        acc_fase_argmax=np.mean(res["fase_arg"]), acc_fase_viterbi=np.mean(res["fase_vit"]), acc_fase_en_vivo=np.mean(res["fase_viv"]),
        f1_segmental_50=np.mean(res["f1_vit"]), conteo_mae=np.mean(res["conteo_abs"]), conteo_obo=np.mean(res["conteo_obo"]),
        conteo_exacto=np.mean(res["conteo_exacto"]), t_concentrica_mae_ms=1000 * np.mean(res["t_con_err"]),
        t_excentrica_mae_ms=1000 * np.mean(res["t_exc_err"]), errores_auc_macro=float(np.nanmean(list(aucs.values()))),
        errores_f1=f1(sc > 0.5), errores_f1_sin_calibrar=f1_crudo, errores_calibrados=calib is not None,
        calidad_mae=np.mean(res["cal_mae"]), fatiga_mae=np.mean(res["fat_mae"]),
        decodificador_mpjpe_mm=np.mean(res["mpjpe_mm"]), decodificador_angulos_mae_grados=np.mean(res["ang_mae"]),
        latencia_ms_por_fotograma=np.mean(res["lat_ms"]), auc_por_error=aucs)
    return {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in resumen.items()}


# ==================================================================== MM-Fit ===
def evaluar_mmfit(modelo, sesiones=BR.MMFIT_TEST + BR.MMFIT_TEST_NUEVOS, modos=("captura", "imu", "ambos")):
    out = {}
    for modo in modos:
        acc_f, cont_abs, cont_obo, n_series, acc_serie = [], [], [], 0, []
        por_ej = {}
        for w in sesiones:
            try:
                s = BR.mmfit_sesion(w)
            except FileNotFoundError:
                continue
            L = s["L"] if modo in ("captura", "ambos") else None
            A = s["angulos"] if modo in ("captura", "ambos") else None
            imu = s["imu"] if modo in ("imu", "ambos") else None
            T = len(s["frames"])
            # por trozos de 3000 fotogramas (100 s) para acotar memoria
            p_ej = np.zeros((T, 17)); dens = np.zeros(T)
            for a in range(0, T, 3000):
                b = min(T, a + 3000)
                cel, P = armar_celdas(A=None if A is None else A[a:b], L=None if L is None else L[a:b],
                                      imu=None if imu is None else imu[a:b].astype(np.float64), T=b - a)
                r = inferir(modelo, cel, P, decodificar=False)
                p_ej[a:b], dens[a:b] = r["p_ej"], r["dens"]
            y = np.where(s["ejercicio"] >= 0, s["ejercicio"], NINGUNO)
            en_serie = s["ejercicio"] >= 0
            acc_f.append(float((p_ej[en_serie].argmax(-1) == y[en_serie]).mean()))
            for k, (a, b, n, nombre) in enumerate(s["series"]):
                m = (s["frames"] >= a) & (s["frames"] <= b)
                c = dens[m].sum()
                cont_abs.append(abs(c - n))
                cont_obo.append(int(abs(round(c) - n) <= 1))
                acierto = int(p_ej[m, :NINGUNO].sum(0).argmax() == NOMBRES_EJ.index(nombre))
                acc_serie.append(acierto)
                por_ej.setdefault(nombre, []).append((acierto, abs(c - n)))
                n_series += 1
        out[modo] = dict(acc_ejercicio_fotograma=float(np.mean(acc_f)), acc_ejercicio_serie=float(np.mean(acc_serie)),
                         conteo_mae=float(np.mean(cont_abs)), conteo_obo=float(np.mean(cont_obo)), series=n_series,
                         por_ejercicio={e: dict(series=len(x), acc_serie=float(np.mean([a for a, _ in x])),
                                                conteo_mae=float(np.mean([c for _, c in x]))) for e, x in por_ej.items()})
        print("MM-Fit", modo, out[modo], flush=True)
    return out


# ====================================================================== EC3D ===
def evaluar_ec3d(modelo, calib=None):
    reps = BR.ec3d_repeticiones()
    aciertos_ej, aciertos_err, aciertos_bin, n = 0, 0, 0, 0
    por_clase = {}
    for r in reps:
        cel, P = armar_celdas(A=r["angulos"], L=r["L"], T=len(r["L"]))
        o = inferir(modelo, cel, P, decodificar=False)
        ej_pred = NOMBRES_EJ[int(o["p_ej"][:, :NINGUNO].sum(0).argmax())]
        aciertos_ej += int(ej_pred == r["ejercicio"])
        cand = [(k, CATALOGO_ERRORES.index(f"{r['ejercicio']}:{k}")) for k in EJERCICIOS[r["ejercicio"]]["errores"]
                if k in ("pies_muy_abiertos", "rodillas_valgo", "poca_profundidad", "tronco_inclinado",
                         "rodilla_pasa_punta", "espalda_arqueada", "espalda_encorvada")]
        pe = {k: float(o["p_err"][:, g].mean()) for k, g in cand}
        if calib is not None:
            pe = {k: float(calibracion.aplicar(calib["repeticion"], CATALOGO_ERRORES[g], pe[k])) for k, g in cand}
        mejor = max(pe, key=pe.get)
        pred = mejor if pe[mejor] > 0.5 else None
        aciertos_err += int(pred == r["error"])
        aciertos_bin += int((pred is None) == (r["error"] is None))
        clave = f"{r['ejercicio']}:{r['error'] or 'correcta'}"
        c = por_clase.setdefault(clave, [0, 0])
        c[0] += int(pred == r["error"]); c[1] += 1
        n += 1
    # EC3D publica cada pose normalizada (pelvis en el origen, columna vertical y de longitud fija): se mide
    # para dejar constancia de que la orientación respecto a la gravedad no está en los datos
    col = {}
    for r in reps:
        v = (r["L"][:, 11] + r["L"][:, 12]) / 2 - (r["L"][:, 23] + r["L"][:, 24]) / 2
        ang = np.degrees(np.arccos(np.clip(v[:, 1] / np.linalg.norm(v, axis=1), -1, 1)))
        col.setdefault(r["ejercicio"], []).append(ang)
    columna = {e: [round(float(x), 1) for x in np.percentile(np.concatenate(a), [50, 95])] for e, a in col.items()}
    return dict(repeticiones=n, acc_ejercicio=aciertos_ej / n, acc_error_exacto=aciertos_err / n,
                acc_correcta_vs_error=aciertos_bin / n, por_clase={k: v[0] / v[1] for k, v in por_clase.items()},
                columna_respecto_vertical_grados_p50_p95=columna)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--mmfit", action="store_true")
    ap.add_argument("--ec3d", action="store_true")
    ap.add_argument("--hilos", type=int, default=8)
    ap.add_argument("--datos", default=None, help="conjunto sintético (por defecto config.CONJUNTO)")
    ap.add_argument("--sin_sintetico", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    modelo = cargar(a.modelo)
    calib = calibracion.cargar(a.modelo)
    dst = Path(a.modelo).with_name("evaluacion.json")
    # se conserva lo evaluado antes (p. ej. MM-Fit) y se reemplaza solo lo que se evalúa ahora
    res = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
    res.update(modelo=a.modelo, parametros=modelo.n_parametros(), activos=modelo.n_activos())
    if not a.sin_sintetico:
        cache = Cache(construir_cache(a.datos))
        _, _, te = cache.particion()
        res["sintetico"] = evaluar_sintetico(modelo, cache, te, a.n, calib=calib)
        print(json.dumps(res["sintetico"], indent=1, ensure_ascii=False), flush=True)
    if a.mmfit:
        res["mmfit"] = evaluar_mmfit(modelo)
    if a.ec3d:
        res["ec3d"] = evaluar_ec3d(modelo, calib=calib)
        print(json.dumps(res["ec3d"], indent=1, ensure_ascii=False))
    dst.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("guardado", dst)


if __name__ == "__main__":
    main()
