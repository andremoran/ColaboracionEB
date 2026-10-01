"""Líneas base clásicas contra las que se mide K3Mov (inspiradas en los proyectos de github-IA).

  SVM   · núcleo RBF sobre estadísticos de ventanas de 3 s de los ángulos (proyectos SVM y KNN).
  GRU   · red recurrente de una capa sobre la celda de ángulos, causal (proyecto de ANN y series
          temporales); misma tarea por fotograma que K3Mov: ejercicio y fase.
  Fase  · contador de repeticiones sin aprendizaje: número de vueltas de la trayectoria en el plano
          de fase (q, q̇) de la primera componente principal de los ángulos (el espacio de fase que
          index3 ya exporta).
Todas usan SOLO los ángulos (lo que index3 exporta en su CSV); K3Mov puede usar además IMU, EMG y
forma corporal. Se evalúan en el conjunto sintético de prueba (sujetos no vistos), en MM-Fit (sesiones
de prueba, datos reales) y en EC3D (repeticiones reales).
Uso: python P1_K3Mov/lineas_base.py [--pasos_gru 1500]
Salida: P1_K3Mov/resultados/lineas_base.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from k3mov.datos import Cache, Ventanas, colar, construir_cache, armar_celdas, NINGUNO  # noqa: E402
from nucleo import bases_reales as BR  # noqa: E402
from nucleo.angulos_clinicos import IDX  # noqa: E402
from nucleo.ejercicios import NOMBRES_EJ  # noqa: E402

DOF = ["hipR_flex", "hipL_flex", "kneeR_flex", "kneeL_flex", "ankleR_flex", "ankleL_flex", "hipR_abd", "hipL_abd",
       "shoulderR_flex", "shoulderL_flex", "shoulderR_abd", "shoulderL_abd", "elbowR_flex", "elbowL_flex",
       "trunk_flex", "trunk_lat", "trunk_rot"]
W, PASO = 90, 30                      # ventana de 3 s, salto de 1 s (30 fps)


# ============================================================ utilidades ===
def _angulos(A):
    X = np.stack([A[:, IDX[k]] for k in DOF], -1).astype(np.float64)
    for k in range(X.shape[1]):                          # huecos -> interpolación; columna vacía -> 0
        ok = np.isfinite(X[:, k])
        X[:, k] = np.interp(np.arange(len(X)), np.flatnonzero(ok), X[ok, k]) if ok.sum() > 1 else 0.0
    return X


def rasgos(X, fps=30.0):
    """136 estadísticos de una ventana de ángulos (T,17)."""
    v = np.gradient(X, 1 / fps, axis=0)
    return np.concatenate([X.mean(0), X.std(0), X.min(0), X.max(0), np.percentile(X, 10, 0), np.percentile(X, 90, 0),
                           np.abs(v).mean(0), v.std(0)]).astype(np.float32)


def ventanas(X, etiqueta=None):
    """Rasgos de ventanas deslizantes; con etiqueta por fotograma devuelve también la mayoritaria."""
    F_, y, centros = [], [], []
    for a in range(0, max(1, len(X) - W + 1), PASO):
        b = min(len(X), a + W)
        F_.append(rasgos(X[a:b]))
        centros.append((a + b) // 2)
        if etiqueta is not None:
            y.append(np.bincount(etiqueta[a:b], minlength=NINGUNO + 1).argmax())
    return np.array(F_), np.array(y), np.array(centros)


PARES = [(DOF.index(f"{a}R_{b}"), DOF.index(f"{a}L_{b}")) for a, b in
         (("hip", "flex"), ("knee", "flex"), ("ankle", "flex"), ("hip", "abd"), ("shoulder", "flex"),
          ("shoulder", "abd"), ("elbow", "flex"))]


def alternancia(v, frac=0.3):
    """Índice en [-1, 1]: +1 si la componente mueve los pares derecha/izquierda en oposición (un lado
    sube mientras el otro baja), -1 si los mueve a la vez. Solo cuentan los pares con carga apreciable
    (los demás son ruido del motor de ángulos), y el par dominante tiene que ser alterno."""
    peso = np.array([abs(v[r]) + abs(v[l]) for r, l in PARES])
    usar = peso >= frac * peso.max()
    dom = int(np.argmax(peso))
    r0, l0 = PARES[dom]
    # el par dominante ha de ser una flexión de miembro (codo, rodilla, cadera): las abducciones y la
    # flexión de hombro del motor de ángulos cambian de signo con el brazo a 90° o tumbado
    if -v[r0] * v[l0] <= 0 or not DOF[r0].startswith(("elbow", "knee", "hipR_flex")):
        return -1.0
    num = sum(-v[r] * v[l] for (r, l), u in zip(PARES, usar) if u)
    den = sum((v[r] ** 2 + v[l] ** 2) / 2 for (r, l), u in zip(PARES, usar) if u) + 1e-12
    return float(num / den)


def contar_por_fase(X, fps=30.0, fc=1.5, r_min=0.35, umbral_alt=0.8):
    """Repeticiones = vueltas completas en el plano de fase (z, ż) de la 1.ª componente principal.
    Solo cuenta el giro cuando el radio supera r_min (las oscilaciones pequeñas del ruido no suman).
    En ejercicios alternados (curl alterno, zancada alterna) cada vuelta son dos repeticiones."""
    Xc = X - X.mean(0)
    if len(Xc) < 10 or not np.isfinite(Xc).all():
        return 0.0
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    x = Xc @ Vt[0]
    b, a = butter(2, fc / (fps / 2))
    x = filtfilt(b, a, x) if len(x) > 12 else x
    z = (x - np.median(x)) / (x.std() + 1e-9)
    v = np.gradient(z)
    y = v / (v.std() + 1e-9)
    th = np.unwrap(np.arctan2(y, z))
    r = np.hypot(z, y)
    d = np.diff(th) * ((r[1:] > r_min) & (r[:-1] > r_min))
    vueltas = float(abs(d.sum()) / (2 * np.pi))
    return 2 * vueltas if alternancia(Vt[0]) > umbral_alt else vueltas


# ================================================================= SVM ===
def entrenar_svm(cache, idx, max_ventanas=16000, semilla=0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    d = cache.d
    Xs, ys = [], []
    for s in idx:
        i0, T = int(d["seq_inicio"][s]), int(d["seq_T"][s])
        A = np.asarray(d["angulos_captura"][i0:i0 + T], np.float64)
        fase = np.asarray(d["fase"][i0:i0 + T])
        et = np.where(fase > 0, int(d["seq_ej"][s]), NINGUNO)
        f, y, _ = ventanas(_angulos(A), et)
        Xs.append(f); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    rng = np.random.default_rng(semilla)
    sel = rng.choice(len(X), size=min(max_ventanas, len(X)), replace=False)
    modelo = make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale"))
    modelo.fit(X[sel], y[sel])
    return modelo


# ================================================================= GRU ===
class GRUBase(nn.Module):
    def __init__(self, d_in=90, h=96):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.ej = nn.Linear(h, NINGUNO + 1)
        self.fase = nn.Linear(h, 5)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.ej(h), self.fase(h)


def entrenar_gru(cache, tr, pasos=1500, semilla=0):
    torch.manual_seed(semilla)
    ds = Ventanas(cache, tr, 128, n_por_epoca=pasos * 16, semilla=semilla)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=colar, num_workers=2, persistent_workers=True)
    red = GRUBase()
    opt = torch.optim.AdamW(red.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=pasos, pct_start=0.05)
    t0 = time.time()
    for paso, lote in enumerate(dl, 1):
        x = lote["celdas"]["ang_cap"]
        v = lote["valido"].float()
        le, lf = red(x)
        loss = ((F.cross_entropy(le.transpose(1, 2), lote["ejercicio"], reduction="none") +
                 F.cross_entropy(lf.transpose(1, 2), lote["fase"], reduction="none")) * v).sum() / v.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(red.parameters(), 1.0); opt.step(); sch.step()
        if paso % 250 == 0:
            print(f"GRU paso {paso} pérdida {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if paso >= pasos:
            break
    return red.eval()


@torch.no_grad()
def gru_serie(red, A, L=None):
    cel, P = armar_celdas(A=A, L=L, T=len(A))
    x = torch.from_numpy(cel["ang_cap"][None].astype(np.float32))
    le, lf = red(x)
    return le[0].softmax(-1).numpy(), lf[0].argmax(-1).numpy()


# ========================================================== evaluación ===
def evaluar_sintetico(cache, te, svm, gru, n_max=300):
    d = cache.d
    acc_svm_v, acc_svm_s, acc_gru_f, acc_gru_fase, acc_gru_s, cnt_err, cnt_obo = [], [], [], [], [], [], []
    for s in te[:n_max]:
        i0, T = int(d["seq_inicio"][s]), int(d["seq_T"][s])
        A = np.asarray(d["angulos_captura"][i0:i0 + T], np.float64)
        fase = np.asarray(d["fase"][i0:i0 + T])
        ej = int(d["seq_ej"][s])
        et = np.where(fase > 0, ej, NINGUNO)
        X = _angulos(A)
        f, y, _ = ventanas(X, et)
        p = svm.predict(f)
        acc_svm_v.append((p == y).mean())
        en = y != NINGUNO
        if en.any():
            acc_svm_s.append(int(np.bincount(p[en], minlength=NINGUNO + 1)[:NINGUNO].argmax() == ej))
        pe, pf = gru_serie(gru, A)
        acc_gru_f.append((pe.argmax(-1) == et).mean())
        acc_gru_fase.append((pf == fase).mean())
        acc_gru_s.append(int(pe[fase > 0, :NINGUNO].sum(0).argmax() == ej) if (fase > 0).any() else 0)
        n_rep = len(cache.ev_por_seq[s])
        activo = fase > 0
        if activo.any():
            a, b = np.flatnonzero(activo)[[0, -1]]
            c = contar_por_fase(X[a:b + 1])
            cnt_err.append(abs(c - n_rep)); cnt_obo.append(int(abs(round(c) - n_rep) <= 1))
    return dict(svm_acc_ventana=float(np.mean(acc_svm_v)), svm_acc_serie=float(np.mean(acc_svm_s)),
                gru_acc_ejercicio_fotograma=float(np.mean(acc_gru_f)), gru_acc_fase=float(np.mean(acc_gru_fase)),
                gru_acc_serie=float(np.mean(acc_gru_s)), fase_conteo_mae=float(np.mean(cnt_err)),
                fase_conteo_obo=float(np.mean(cnt_obo)), series=len(acc_svm_v))


def evaluar_mmfit(svm, gru, sesiones=BR.MMFIT_TEST + BR.MMFIT_TEST_NUEVOS):
    acc_svm_f, acc_svm_s, acc_gru_f, acc_gru_s, cnt_err, cnt_obo = [], [], [], [], [], []
    for w in sesiones:
        try:
            s = BR.mmfit_sesion(w, con_imu=False)
        except FileNotFoundError:
            continue
        X = _angulos(s["angulos"])
        y = np.where(s["ejercicio"] >= 0, s["ejercicio"], NINGUNO)
        f, _, centros = ventanas(X)
        p = svm.predict(f)
        pf = np.interp(np.arange(len(X)), centros, p).round().astype(int)       # etiqueta de la ventana más cercana
        en = s["ejercicio"] >= 0
        acc_svm_f.append((pf[en] == y[en]).mean())
        pe = np.zeros((len(X), NINGUNO + 1))
        for a in range(0, len(X), 3000):
            b = min(len(X), a + 3000)
            pe[a:b], _ = gru_serie(gru, s["angulos"][a:b])
        acc_gru_f.append((pe[en].argmax(-1) == y[en]).mean())
        for (a, b, n, nombre) in s["series"]:
            m = (s["frames"] >= a) & (s["frames"] <= b)
            ej = NOMBRES_EJ.index(nombre)
            acc_svm_s.append(int(np.bincount(pf[m], minlength=NINGUNO + 1)[:NINGUNO].argmax() == ej))
            acc_gru_s.append(int(pe[m, :NINGUNO].sum(0).argmax() == ej))
            c = contar_por_fase(X[m])
            cnt_err.append(abs(c - n)); cnt_obo.append(int(abs(round(c) - n) <= 1))
    return dict(svm_acc_fotograma=float(np.mean(acc_svm_f)), svm_acc_serie=float(np.mean(acc_svm_s)),
                gru_acc_fotograma=float(np.mean(acc_gru_f)), gru_acc_serie=float(np.mean(acc_gru_s)),
                fase_conteo_mae=float(np.mean(cnt_err)), fase_conteo_obo=float(np.mean(cnt_obo)), series=len(cnt_err))


def evaluar_ec3d(svm, gru):
    reps = BR.ec3d_repeticiones()
    a_svm, a_gru = [], []
    for r in reps:
        ej = NOMBRES_EJ.index(r["ejercicio"])
        X = _angulos(r["angulos"])
        p = svm.predict(rasgos(X)[None])[0]
        a_svm.append(int(p == ej))
        pe, _ = gru_serie(gru, r["angulos"], r["L"])
        a_gru.append(int(pe[:, :NINGUNO].sum(0).argmax() == ej))
    return dict(svm_acc_repeticion=float(np.mean(a_svm)), gru_acc_repeticion=float(np.mean(a_gru)), repeticiones=len(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasos_gru", type=int, default=1500)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--datos", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    cache = Cache(construir_cache(a.datos))
    tr, va, te = cache.particion()
    t0 = time.time()
    svm = entrenar_svm(cache, tr)
    print(f"SVM entrenada ({time.time() - t0:.0f}s)", flush=True)
    gru = entrenar_gru(cache, tr, a.pasos_gru)
    res = dict(gru_parametros=sum(p.numel() for p in gru.parameters()))
    res["sintetico"] = evaluar_sintetico(cache, te, svm, gru)
    print("sintético", res["sintetico"], flush=True)
    res["mmfit"] = evaluar_mmfit(svm, gru)
    print("MM-Fit", res["mmfit"], flush=True)
    res["ec3d"] = evaluar_ec3d(svm, gru)
    print("EC3D", res["ec3d"], flush=True)
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "lineas_base.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    torch.save(gru.state_dict(), AQUI / "resultados" / "gru_base.pt")


if __name__ == "__main__":
    main()
