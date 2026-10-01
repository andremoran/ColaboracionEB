"""Entrena las redes del Proyecto 2 sobre el conjunto sintético.

  --modelo tcn      B · de la captura ruidosa a momentos (N·m/kg), reacciones (peso) y potencias (W/kg)
  --modelo estado   C · de la captura ruidosa al estado cinemático limpio (para el Newton–Euler exacto)
  --causal          versión en vivo (solo pasado)
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

from dinv.datos import abrir_cache, Ventanas, colar  # noqa: E402
from dinv.modelos import TCN  # noqa: E402
from nucleo.rotaciones import t_seis_d_a_matriz  # noqa: E402

F_IN = 33 * 3 * 2 + 9 + 4


def particion(n, semilla=0):
    idx = np.random.default_rng(semilla).permutation(n)
    nv, nt = int(n * 0.08), int(n * 0.10)
    return idx[nt + nv:], idx[nt:nt + nv], idx[:nt]


def perdida(modelo, lote, tipo):
    out = modelo(lote["X"])
    v = lote["valido"].float()[..., None]
    nv = v.sum().clamp(min=1)
    if tipo == "tcn":
        lm = (((out["M"] - lote["M"]) ** 2) * v).sum() / (nv * 24)
        mf = lote["mF"][..., None] * v[..., None]
        lf = (((out["F"].view(*out["F"].shape[:2], 2, 3) - lote["F"]) ** 2) * mf).sum() / (mf.sum() * 3).clamp(min=1)
        lp = (((out["Pw"] - lote["Pw"]) ** 2) * v).sum() / (nv * 22)
        return lm + 2 * lf + 0.3 * lp, dict(M=lm.item(), F=lf.item(), Pw=lp.item())
    # estado: orientaciones globales 6D + pelvis relativa (en estaturas)
    g6 = out["G6"].view(*out["G6"].shape[:2], 22, 6)
    Rp = t_seis_d_a_matriz(g6)
    Rt = t_seis_d_a_matriz(lote["G6"])
    lg = (((Rp - Rt) ** 2).sum((-1, -2)).mean(-1, keepdim=True) * v).sum() / nv
    P = lote["P"][:, :, 0] - lote["P"][:, :1, 0]
    Pn = P / lote["estatura"][:, None, None]
    lr = (((out["raiz"] - Pn) ** 2).sum(-1, keepdim=True) * v).sum() / nv
    lc = (torch.nn.functional.binary_cross_entropy_with_logits(out["contacto"], lote["contacto"], reduction="none") * v).sum() / (nv * 2)
    return lg + 20 * lr + 0.2 * lc, dict(G=lg.item(), raiz=lr.item(), contacto=lc.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["tcn", "estado"], default="tcn")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--canales", type=int, default=96)
    ap.add_argument("--niveles", type=int, default=5)
    ap.add_argument("--pasos", type=int, default=3000)
    ap.add_argument("--lote", type=int, default=24)
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    torch.manual_seed(0)
    z = abrir_cache()
    tr, va, te = particion(len(z["seq_T"]))
    nombre = f"{a.modelo}{'_vivo' if a.causal else ''}_c{a.canales}"
    sal = AQUI / "modelos" / nombre
    sal.mkdir(parents=True, exist_ok=True)
    salidas = dict(M=24, F=6, Pw=22) if a.modelo == "tcn" else dict(G6=22 * 6, raiz=3, contacto=2)
    modelo = TCN(F_IN, salidas, a.canales, a.niveles, causal=a.causal)
    if a.modelo == "estado":
        with torch.no_grad():
            modelo.cabezas["G6"].bias.copy_(torch.tensor([1.0, 0, 0, 0, 1, 0]).repeat(22))
    ds = Ventanas(z, tr, 96, n=a.pasos * a.lote)
    dv = Ventanas(z, va, 96, n=240, semilla=7)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.lote, collate_fn=colar, num_workers=0)
    opt = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.pasos, pct_start=0.05)
    print(nombre, "parámetros", sum(p.numel() for p in modelo.parameters()), flush=True)
    t0, hist, mejor = time.time(), [], 1e9
    for paso, lote in enumerate(dl, 1):
        l, info = perdida(modelo, lote, a.modelo)
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step(); sch.step()
        if paso % 100 == 0:
            print(f"paso {paso} {l.item():.4f} {info} {time.time() - t0:.0f}s", flush=True)
        if paso % 500 == 0 or paso == a.pasos:
            modelo.eval()
            with torch.no_grad():
                vals = [perdida(modelo, colar([dv[i] for i in range(k, k + 24)]), a.modelo)[0].item() for k in range(0, 240, 24)]
            modelo.train()
            ev = dict(paso=paso, val=float(np.mean(vals)), s=round(time.time() - t0))
            hist.append(ev)
            print("VAL", ev, flush=True)
            if ev["val"] < mejor:
                mejor = ev["val"]
                torch.save(dict(cfg=dict(f_in=F_IN, salidas=salidas, c=a.canales, niveles=a.niveles, causal=a.causal),
                                estado=modelo.state_dict()), sal / "mejor.pt")
        if paso >= a.pasos:
            break
    (sal / "historial.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
