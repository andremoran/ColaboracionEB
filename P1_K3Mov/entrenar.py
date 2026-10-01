"""Entrenamiento de K3Mov sobre el conjunto sintético (preentrenamiento multitarea).

Uso:  python P1_K3Mov/entrenar.py --d 64 --capas 8 --pasos 6000 --nombre k3mov_d64
Pérdidas: ejercicio (CE), fase (CE), densidad de repetición (MSE + conteo por ventana), errores de
técnica (BCE ponderada), calidad y fatiga (MSE), protención MTP (L1), reconstrucción de celdas
ocultas (L1) y decodificador del movimiento correcto (6D + marcas por cinemática directa + ángulos).
Optimizador: Muon por cabeza para las matrices ocultas + AdamW para el resto; recorte QK en MLA.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from k3mov.datos import Cache, Ventanas, colar, construir_cache, N_ERR  # noqa: E402
from k3mov.modelo import K3Mov, NOMBRES_CELDAS  # noqa: E402
from k3mov.optim import Muon, repartir_parametros  # noqa: E402
from nucleo.fk_torch import FKTorch  # noqa: E402

PESOS = dict(ej=1.0, ej_v=0.5, fase=1.0, rep=30.0, conteo=0.3, err=2.0, cal=0.5, fat=0.5, mtp=2.0, rec=0.5,
             rot=1.0, pos=4.0, ang=1.0, raiz=2.0)


def perdidas(modelo, lote, fk, pesos=PESOS, decodificar=True):
    cel, P = lote["celdas"], lote["presentes"]
    oc = lote["oculta"]
    P_in = P & ~oc
    out = modelo(cel, P_in, causal=True, decodificar=decodificar)
    v = lote["valido"].float()
    nv = v.sum().clamp(min=1)
    L = {}
    L["ej"] = (F.cross_entropy(out["ejercicio"].transpose(1, 2), lote["ejercicio"], reduction="none", label_smoothing=0.05) * v).sum() / nv
    L["fase"] = (F.cross_entropy(out["fase"].transpose(1, 2), lote["fase"], reduction="none") * v).sum() / nv
    # ejercicio a nivel de ventana (promedio de h en los fotogramas en ejercicio): señal más fuerte al inicio
    act = v * (lote["ejercicio"] < 16).float()
    hay = act.sum(1) > 0
    if hay.any():
        hm = (out["h"] * act[..., None]).sum(1) / act.sum(1, keepdim=True).clamp(min=1)
        et = torch.where(lote["ejercicio"] < 16, lote["ejercicio"], torch.full_like(lote["ejercicio"], -1)).max(1).values
        L["ej_v"] = F.cross_entropy(modelo.h_ej(hm[hay]), et[hay])
    else:
        L["ej_v"] = torch.zeros((), device=v.device)
    L["rep"] = ((out["rep"] - lote["rep"]) ** 2 * v).sum() / nv
    L["conteo"] = ((out["rep"] * v).sum(1) - (lote["rep"] * v).sum(1)).abs().mean()
    peso_err = 0.2 + lote["mask_err"][:, None, :]                           # errores del ejercicio pesan más
    bce = F.binary_cross_entropy_with_logits(out["errores"], lote["errores"], reduction="none")
    L["err"] = (bce * peso_err * v[..., None]).sum() / (nv * N_ERR * 0.3)
    en_rep = v * (lote["fase"] > 0).float()
    L["cal"] = ((out["calidad"] - lote["calidad"]) ** 2 * en_rep).sum() / en_rep.sum().clamp(min=1)
    L["fat"] = ((out["fatiga"] - lote["fatiga"]) ** 2 * v).sum() / nv
    m = lote["mtp_m"] * v[..., None, None]
    L["mtp"] = ((out["mtp"] - lote["mtp"]).abs() * m).sum() / m.sum().clamp(min=1)
    rec = modelo.reconstruir(out["h"])
    num, den = 0.0, 0.0
    for i, k in enumerate(NOMBRES_CELDAS):
        w = (oc[..., i].float() * v)
        if w.sum() > 0:
            num = num + ((rec[k] - cel[k]).abs().mean(-1) * w).sum()
            den = den + w.sum()
    L["rec"] = num / max(den, 1.0) if den > 0 else torch.zeros((), device=v.device)
    if decodificar:
        d = out["dec"]
        L["rot"] = (((d["rot6"] - lote["rot6"]) ** 2).mean((-1, -2)) * v).sum() / nv
        marcas = fk(d["rot6"], lote["estatura"])
        obj = lote["L_corr"]
        cen_p = marcas[:, :, [23, 24]].mean(2, keepdim=True)
        cen_o = obj[:, :, [23, 24]].mean(2, keepdim=True)
        L["pos"] = ((marcas - cen_p - (obj - cen_o)).norm(dim=-1).mean(-1) * v).sum() / nv
        ma = lote["ang_corr_m"] * v[..., None]
        L["ang"] = ((d["angulos"] - lote["ang_corr"]).abs() * ma).sum() / ma.sum().clamp(min=1)
        L["raiz"] = ((d["raiz"] - lote["raiz"]).abs().mean(-1) * v).sum() / nv
    total = sum(pesos[k] * val for k, val in L.items())
    return total, L, out


@torch.no_grad()
def evaluar(modelo, cargador, fk, n_lotes=20):
    modelo.eval()
    acc_ej = acc_fase = n = 0.0
    err_cont = 0.0
    mpjpe = 0.0
    k = 0
    for lote in cargador:
        _, L, out = perdidas(modelo, lote, fk)
        v = lote["valido"]
        acc_ej += ((out["ejercicio"].argmax(-1) == lote["ejercicio"]) & v).sum().item()
        acc_fase += ((out["fase"].argmax(-1) == lote["fase"]) & v).sum().item()
        n += v.sum().item()
        err_cont += L["conteo"].item()
        mpjpe += L["pos"].item()
        k += 1
        if k >= n_lotes:
            break
    modelo.train()
    return dict(acc_ejercicio=acc_ej / n, acc_fase=acc_fase / n, err_conteo_ventana=err_cont / k, mpjpe_m=mpjpe / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--capas", type=int, default=8)
    ap.add_argument("--dk", type=int, default=16)
    ap.add_argument("--capas_dec", type=int, default=2)
    ap.add_argument("--expertos", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--pasos", type=int, default=6000)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--ventana", type=int, default=160)
    ap.add_argument("--lr_muon", type=float, default=1.5e-3)     # con la escala 0.2·sqrt(max) el RMS iguala al de AdamW
    ap.add_argument("--lr_adam", type=float, default=1.5e-3)
    ap.add_argument("--hilos", type=int, default=12)
    ap.add_argument("--nombre", default="k3mov_d64")
    ap.add_argument("--datos", default=None)          # None -> config.CONJUNTO
    ap.add_argument("--frac_datos", type=float, default=1.0)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--reanudar", action="store_true", help="continúa desde el último punto de control")
    a = ap.parse_args()
    torch.manual_seed(a.semilla)
    torch.set_num_threads(a.hilos)
    salida = AQUI / "modelos" / a.nombre
    salida.mkdir(parents=True, exist_ok=True)
    cache = Cache(construir_cache(a.datos))
    tr, va, te = cache.particion()
    if a.frac_datos < 1.0:
        tr = tr[: max(16, int(len(tr) * a.frac_datos))]
    punto = salida / "punto.pt"
    inicio = 0
    estado_prev = None
    if a.reanudar and punto.exists():
        estado_prev = torch.load(punto, weights_only=False)
        inicio = int(estado_prev["paso"])
        print(f"reanudando desde el paso {inicio}", flush=True)
    # las ventanas del tramo que falta llevan otra semilla: no se repiten las ya vistas
    ds_tr = Ventanas(cache, tr, a.ventana, n_por_epoca=max(1, a.pasos - inicio) * a.lote, semilla=a.semilla + inicio)
    ds_va = Ventanas(cache, va, a.ventana, n_por_epoca=400, semilla=123, aumentar=False)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.lote, collate_fn=colar, num_workers=3,
                                        persistent_workers=True, prefetch_factor=4)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=a.lote, collate_fn=colar, num_workers=1)
    modelo = K3Mov(d=a.d, capas=a.capas, dk=a.dk, n_exp=a.expertos, k=a.topk, n_err=N_ERR, capas_dec=a.capas_dec)
    fk = FKTorch()
    muon_p, adam_p = repartir_parametros(modelo)
    opt_m = Muon(muon_p, lr=a.lr_muon)
    opt_a = torch.optim.AdamW(adam_p, lr=a.lr_adam, weight_decay=0.01)
    hist, mejor = [], -1
    if estado_prev is not None:
        modelo.load_state_dict(estado_prev["estado"])
        opt_m.load_state_dict(estado_prev["opt_m"])
        opt_a.load_state_dict(estado_prev["opt_a"])
        hist, mejor = estado_prev["hist"], estado_prev["mejor"]
    print(f"parametros {modelo.n_parametros():,}  activos {modelo.n_activos():,}  muon={len(muon_p)} adam={len(adam_p)}", flush=True)
    t0 = time.time()
    for paso, lote in enumerate(dl_tr, inicio + 1):
        frac = paso / a.pasos
        esc = min(1.0, paso / 200) * (0.5 * (1 + math.cos(math.pi * min(frac, 1.0))) * 0.9 + 0.1)
        for g in opt_m.param_groups:
            g["lr"] = a.lr_muon * esc
        for g in opt_a.param_groups:
            g["lr"] = a.lr_adam * esc
        total, L, _ = perdidas(modelo, lote, fk)
        opt_m.zero_grad(); opt_a.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(adam_p, 1.0)
        opt_m.step(); opt_a.step()
        for mla in modelo.capas_mla():
            mla.recorte_qk(30.0)
        if paso % 50 == 0:
            msg = {k: round(float(v), 4) for k, v in L.items()}
            print(f"paso {paso} total {total.item():.3f} {msg} {time.time() - t0:.0f}s", flush=True)
        if paso % 500 == 0 or paso == a.pasos:
            ev = evaluar(modelo, dl_va, fk)
            ev.update(paso=paso, segundos=round(time.time() - t0))
            hist.append(ev)
            print("VAL", ev, flush=True)
            puntaje = ev["acc_ejercicio"] + ev["acc_fase"] - ev["mpjpe_m"]
            if puntaje > mejor:
                mejor = puntaje
                torch.save({"cfg": modelo.cfg, "estado": modelo.state_dict()}, salida / "mejor.pt")
            (salida / "historial.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
        if paso % 250 == 0:                               # punto de control completo (reanudable)
            torch.save({"cfg": modelo.cfg, "estado": modelo.state_dict(), "opt_m": opt_m.state_dict(),
                        "opt_a": opt_a.state_dict(), "paso": paso, "hist": hist, "mejor": mejor}, punto)
        if paso >= a.pasos:
            break
    torch.save({"cfg": modelo.cfg, "estado": modelo.state_dict()}, salida / "ultimo.pt")
    print("fin", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
