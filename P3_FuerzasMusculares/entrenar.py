"""Redes del Proyecto 3: de ángulos, velocidades y momentos articulares a las activaciones de los 80
músculos de MyoLeg, en un solo paso (optimización amortizada).

  --modo supervisado : aprende a imitar la optimización estática (destilación).
  --modo fisico      : SIN etiquetas de activación; minimiza el residuo de la ecuación de momentos
                       -Σ r_ij (a_i Fa_i + Fp_i) = τ_j más el esfuerzo Σ a_i² sobre la geometría
                       muscular de MuJoCo guardada por fotograma (red informada por la física).
Ambas se comparan con la optimización estática en fotogramas de sujetos no vistos.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from nucleo import config  # noqa: E402


class RedMuscular(nn.Module):
    def __init__(self, n_dof=12, n_mus=80, oculto=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * n_dof + 2, oculto), nn.GELU(), nn.Linear(oculto, oculto), nn.GELU(),
                                 nn.Linear(oculto, oculto), nn.GELU(), nn.Linear(oculto, n_mus))

    def forward(self, q, qd, tau, masa):
        x = torch.cat([torch.sin(q), torch.cos(q), qd / 5.0, tau / (masa[:, None] * 1.5), masa[:, None] / 70.0,
                       torch.ones_like(masa[:, None])], -1)
        return torch.sigmoid(self.net(x) - 2.0)


def cargar():
    z = np.load(config.DATOS_GEN / config.SO_MYOLEG)
    d = {k: z[k] for k in z.files}
    n = len(d["seq_off"])
    idx = np.random.default_rng(0).permutation(n)
    te = set(idx[: int(n * 0.12)].tolist())
    frame_seq = np.concatenate([np.full(l, s) for s, (o, l) in enumerate(d["seq_off"])])
    masa = np.concatenate([np.full(l, d["seq_sujeto"][s][1]) for s, (o, l) in enumerate(d["seq_off"])])
    es_te = np.isin(frame_seq, list(te))
    return d, masa.astype(np.float32), es_te


def residuo_fisico(a, R, Fa, Fp, tau):
    F = a * Fa + Fp
    return -(R * F[..., None]).sum(1) - tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["supervisado", "fisico"], default="supervisado")
    ap.add_argument("--pasos", type=int, default=6000)
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    torch.manual_seed(0)
    d, masa, es_te = cargar()
    geo = d["geo_pos"]
    geo_tr = np.flatnonzero(~es_te[geo])
    geo_te = np.flatnonzero(es_te[geo])
    T = lambda x: torch.tensor(np.asarray(x, dtype=np.float32))
    q, qd, tau, A = T(d["q"]), T(d["qd"]), T(d["tau"]), T(d["a"].astype(np.float32))
    R, Fa, Fp = T(d["R"].astype(np.float32)), T(d["Fa"]), T(d["Fp"])
    M = T(masa)
    red = RedMuscular()
    opt = torch.optim.AdamW(red.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.pasos)
    tr_frames = np.flatnonzero(~es_te)
    t0 = time.time()
    for paso in range(1, a.pasos + 1):
        if a.modo == "supervisado":
            i = torch.from_numpy(np.random.choice(tr_frames, 512))
            pred = red(q[i], qd[i], tau[i], M[i])
            loss = ((pred - A[i]) ** 2).mean()
        else:
            gi = np.random.choice(geo_tr, 512)
            i = torch.from_numpy(geo[gi])
            pred = red(q[i], qd[i], tau[i], M[i])
            r = residuo_fisico(pred, R[gi], Fa[gi], Fp[gi], tau[i]) / (M[i][:, None] * 0.05)
            loss = (r ** 2).mean() + 0.02 * (pred ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if paso % 1000 == 0:
            print(f"paso {paso} pérdida {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    # evaluación en geometría de sujetos no vistos
    red.eval()
    with torch.no_grad():
        i = torch.from_numpy(geo[geo_te])
        pred = red(q[i], qd[i], tau[i], M[i])
        res_red = residuo_fisico(pred, R[geo_te], Fa[geo_te], Fp[geo_te], tau[i])
        res_so = residuo_fisico(A[i], R[geo_te], Fa[geo_te], Fp[geo_te], tau[i])
        err_a = (pred - A[i]).abs()
        F_red = pred * Fa[geo_te] + Fp[geo_te]
        F_so = A[i] * Fa[geo_te] + Fp[geo_te]
        corr = np.corrcoef(F_red.numpy().ravel(), F_so.numpy().ravel())[0, 1]
        t1 = time.time()
        for _ in range(20):
            red(q[i[:1]], qd[i[:1]], tau[i[:1]], M[i[:1]])
        ms = (time.time() - t1) / 20 * 1000
    out = dict(modo=a.modo, parametros=sum(p.numel() for p in red.parameters()),
               residuo_momento_rmse_Nm_red=float(res_red.pow(2).mean().sqrt()),
               residuo_momento_rmse_Nm_optim_estatica=float(res_so.pow(2).mean().sqrt()),
               activacion_mae_vs_SO=float(err_a.mean()), fuerza_corr_vs_SO=float(corr),
               esfuerzo_red=float((pred ** 2).sum(-1).mean()), esfuerzo_SO=float((A[i] ** 2).sum(-1).mean()),
               ms_por_fotograma_red=ms, fotogramas_prueba=int(len(geo_te)))
    sal = AQUI / "modelos"
    sal.mkdir(exist_ok=True)
    torch.save(red.state_dict(), sal / f"red_{a.modo}.pt")
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / f"red_{a.modo}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
