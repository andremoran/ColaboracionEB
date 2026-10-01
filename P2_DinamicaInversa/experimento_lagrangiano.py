"""Experimento lagrangiano: ¿generaliza mejor una red con la estructura de Euler–Lagrange?

Cadena sagital con el pie fijo al suelo (tobillo, rodilla, cadera, lumbar), masas e inercias de de
Leva para 75 kg y 1.75 m (las dos piernas sumadas, cabeza-brazos-tronco como un segmento). La verdad
de terreno la da MuJoCo (mj_inverse). Se entrena con movimientos ALEATORIOS suaves y se prueba con
patrones de ejercicio nunca vistos (sentadilla, peso muerto rumano, elevación de talones, sentadilla
rápida). Se comparan: DeLaN (M(q) = L Lᵀ, V(q), Coriolis por diferenciación automática) y un
perceptrón sin estructura con los mismos datos.
Salida: resultados/lagrangiano.json
"""
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

from dinv.modelos import DeLaN, PerceptronID  # noqa: E402
from nucleo.antropometria import DE_LEVA  # noqa: E402

MASA, EST = 75.0, 1.75


def mjcf():
    f = lambda k: DE_LEVA[k]["H"]
    Lp = 0.246 * EST        # pierna (rodilla-tobillo)
    Lm = 0.245 * EST        # muslo
    Lt = 0.288 * EST        # tronco (cadera-hombro) + cabeza y brazos como un segmento
    def cuerpo(m, L, cm, r):
        return m, L, cm * L, m * (r * L) ** 2
    mp, _, cp, Ip = cuerpo(2 * MASA * f("pierna")[0] / 100, Lp, 1 - f("pierna")[1] / 100, f("pierna")[2] / 100)
    mm, _, cmm, Im = cuerpo(2 * MASA * f("muslo")[0] / 100, Lm, 1 - f("muslo")[1] / 100, f("muslo")[2] / 100)
    m_hat = MASA * (f("cabeza")[0] + f("torax")[0] + f("abdomen")[0] + 2 * (f("brazo")[0] + f("antebrazo")[0] + f("mano")[0])) / 100
    m_pel = MASA * f("pelvis")[0] / 100
    I_hat = m_hat * (0.3 * Lt * 1.6) ** 2
    return f"""<mujoco><option gravity="0 0 -9.81"/><worldbody>
  <body name="pierna" pos="0 0 0.08">
    <joint name="tobillo" type="hinge" axis="0 1 0"/>
    <inertial pos="0 0 {cp:.5f}" mass="{mp:.5f}" diaginertia="{Ip:.6f} {Ip:.6f} {Ip*0.1:.6f}"/>
    <body name="muslo" pos="0 0 {Lp:.5f}">
      <joint name="rodilla" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {cmm:.5f}" mass="{mm:.5f}" diaginertia="{Im:.6f} {Im:.6f} {Im*0.1:.6f}"/>
      <body name="pelvis" pos="0 0 {Lm:.5f}">
        <joint name="cadera" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0.05" mass="{m_pel:.5f}" diaginertia="0.08 0.08 0.05"/>
        <body name="tronco" pos="0 0 0.10">
          <joint name="lumbar" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 {0.62*Lt:.5f}" mass="{m_hat:.5f}" diaginertia="{I_hat:.5f} {I_hat:.5f} {I_hat*0.3:.5f}"/>
        </body></body></body></body></worldbody></mujoco>"""


def verdad_terreno(m, Q, fs):
    d = mujoco.MjData(m)
    dt = 1 / fs
    qd = np.gradient(Q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    tau = np.zeros_like(Q)
    for i in range(len(Q)):
        d.qpos[:], d.qvel[:], d.qacc[:] = Q[i], qd[i], qdd[i]
        mujoco.mj_inverse(m, d)
        tau[i] = d.qfrc_inverse
    return qd, qdd, tau


RANGO = np.array([[-0.35, 0.6], [-2.1, 0.05], [-0.1, 1.9], [-0.25, 0.5]])   # tobillo, rodilla, cadera, lumbar


def aleatorio(rng, T, fs, fmax=0.45):
    """Movimientos aleatorios suaves que recorren todo el rango articular, pero LENTOS (f <= fmax)."""
    t = np.arange(T) / fs
    Q = np.zeros((T, 4))
    for j in range(4):
        x = np.zeros(T)
        for k in range(5):
            f = rng.uniform(0.05, fmax)
            x += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * rng.uniform(0.3, 1.0)
        x = (x - x.min()) / (x.max() - x.min() + 1e-9)
        Q[:, j] = RANGO[j, 0] + x * (RANGO[j, 1] - RANGO[j, 0])
    return Q


def patrones(fs):
    t = np.arange(int(8 * fs)) / fs
    s = lambda f: 0.5 - 0.5 * np.cos(2 * np.pi * f * t)
    out = {}
    # ángulos en rad: tobillo (+ adelante), rodilla (- flexión), cadera (+ flexión), lumbar (+ flexión)
    out["sentadilla"] = np.stack([0.45 * s(0.35), -1.9 * s(0.35), 1.6 * s(0.35), 0.15 * s(0.35)], -1)
    out["sentadilla_rapida_x2"] = np.stack([0.45 * s(0.9), -1.9 * s(0.9), 1.6 * s(0.9), 0.15 * s(0.9)], -1)
    out["peso_muerto"] = np.stack([0.1 * s(0.3), -0.35 * s(0.3), 1.3 * s(0.3), 0.1 * s(0.3)], -1)
    out["talones_rapido"] = np.stack([-0.25 * s(1.0), 0 * t, 0 * t, 0 * t], -1)
    return out


def entrenar(modelo, datos, pasos=8000, lr=3e-3, lote=256, semilla=0):
    torch.manual_seed(semilla)
    q, qd, qdd, tau = [torch.tensor(x, dtype=torch.float32) for x in datos]
    esc = tau.std(0) + 1e-6
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, pasos)
    for p in range(pasos):
        i = torch.randint(0, len(q), (lote,))
        pred, _ = modelo(q[i], qd[i], qdd[i])
        loss = (((pred - tau[i]) / esc) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return modelo


def evaluar(modelo, datos):
    q, qd, qdd, tau = [torch.tensor(x, dtype=torch.float32) for x in datos]
    pred = []
    for a in range(0, len(q), 512):
        p, _ = modelo(q[a:a + 512], qd[a:a + 512], qdd[a:a + 512])
        pred.append(p.detach())
    pred = torch.cat(pred)
    err = (pred - tau).numpy()
    rmse = np.sqrt((err ** 2).mean(0))
    return dict(rmse_Nm=rmse.tolist(), rmse_Nm_kg=(rmse / MASA).tolist(),
                nrmse=float(np.sqrt((err ** 2).mean()) / (tau.numpy().std() + 1e-9)))


def main():
    torch.set_num_threads(4)
    fs = 100.0
    m = mujoco.MjModel.from_xml_string(mjcf())
    rng = np.random.default_rng(0)
    Qtr = np.concatenate([aleatorio(rng, int(60 * fs), fs) for _ in range(5)])     # 5 minutos de movimiento lento
    datos_tr = (Qtr,) + verdad_terreno(m, Qtr, fs)
    pruebas = {k: (Q,) + verdad_terreno(m, Q, fs) for k, Q in patrones(fs).items()}
    Qva = aleatorio(np.random.default_rng(9), int(30 * fs), fs)
    pruebas["aleatorio_nuevo"] = (Qva,) + verdad_terreno(m, Qva, fs)
    res = {}
    for nombre, cls in (("DeLaN", lambda: DeLaN(4, 64)), ("Perceptron", lambda: PerceptronID(4, 128))):
        t0 = time.time()
        mod = entrenar(cls(), datos_tr)
        res[nombre] = dict(segundos=round(time.time() - t0, 1), parametros=sum(p.numel() for p in mod.parameters()),
                           pruebas={k: evaluar(mod, v) for k, v in pruebas.items()})
        if nombre == "DeLaN":
            # comprobación de estructura: M(q) definida positiva y balance de energía en la sentadilla
            q, qd, qdd, tau = [torch.tensor(x, dtype=torch.float32) for x in pruebas["sentadilla"]]
            Tk, Vq = mod.energia(q, qd)
            H = (Tk + Vq).detach().numpy()
            pot = (tau * qd).sum(-1).numpy()                   # potencia que entra por las articulaciones
            dH = np.gradient(H, 1 / fs)
            res[nombre]["balance_energia_r"] = float(np.corrcoef(dH[5:-5], pot[5:-5])[0, 1])
            eig = torch.linalg.eigvalsh(mod.masa(q[:200])).min().item()
            res[nombre]["min_autovalor_M"] = eig
        print(nombre, json.dumps(res[nombre], indent=None)[:600], flush=True)
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "lagrangiano.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
