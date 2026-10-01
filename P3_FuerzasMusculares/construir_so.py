"""Optimización estática (MyoLeg, 80 músculos) sobre el conjunto sintético, en paralelo.

Para cada secuencia (1 de cada 3 fotogramas, 10 Hz): coordenadas de MyoLeg desde las rotaciones del
avatar, brazos de momento y capacidades (MuJoCo), momentos generalizados desde la dinámica inversa
(Newton–Euler validado) y activaciones/fuerzas por optimización estática. En una muestra de
fotogramas se guarda también la geometría (brazos, capacidades) para la pérdida física de la red.
Salida: datos_generados/<config.SO_MYOLEG> (so_myoleg_v3.npz)
"""
import glob
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))

PASO = 3


def trabajo(args):
    ruta, i = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from musculos.myoleg import Musculos, coordenadas, qpos, qvel, DOFS, fuerzas_generalizadas
    from musculos.optimizacion import resolver
    from nucleo.ejercicios import NOMBRES_EJ
    z = np.load(ruta)
    g = lambda k: z[f"{i}__{k}"]
    R6 = g("R6").astype(np.float64)[::PASO]
    mom = g("momentos").astype(np.float64)[::PASO]
    if not np.isfinite(R6).all() or not np.isfinite(mom).all():
        return None
    mus = Musculos()
    q = coordenadas(R6)
    Q = qpos(mus.m, q)
    Qd = qvel(mus.m, Q, 30.0 / PASO)
    R, Lm, Vm = mus.geometria(Q, Qd)
    Fa, Fp = mus.capacidad(Lm, Vm)
    tau = fuerzas_generalizadas(R6, mom)              # proyección exacta sobre los ejes de MyoLeg
    a, F, res = resolver(R, Fa, Fp, tau)
    qv = np.stack([q[d] for d in DOFS], -1)
    rng = np.random.default_rng(int(g("semilla")))
    geo = rng.random(len(Q)) < 0.25
    return dict(ej=int(g("ejercicio")), sujeto=g("sujeto"), q=qv.astype(np.float32),
                qd=np.gradient(qv, PASO / 30.0, axis=0).astype(np.float32), tau=tau.astype(np.float32),
                a=a.astype(np.float16), F=F.astype(np.float32), res=res.astype(np.float32),
                fase=g("fase")[::PASO].astype(np.int8), geo_idx=np.flatnonzero(geo).astype(np.int32),
                R=R[geo].astype(np.float16), Fa=Fa[geo].astype(np.float32), Fp=Fp[geo].astype(np.float32),
                semilla=int(g("semilla")))


def trabajo_a_disco(args):
    """Resuelve una secuencia y guarda el resultado en su propio archivo (reanudable)."""
    ruta, i, parte = args
    if Path(parte).exists():
        return 1
    r = trabajo((ruta, i))
    if r is None:
        Path(parte + ".vacia").write_text("", encoding="utf-8")
        return 0
    tmp = parte + ".tmp.npz"
    np.savez(tmp, **{k: np.asarray(v) for k, v in r.items()})
    Path(tmp).replace(parte)
    return 1


def main(n_proc=10, max_seq=None, nombre=None, salida=None):
    from nucleo import config
    nombre, salida = nombre or config.CONJUNTO_DIN, salida or config.SO_MYOLEG
    partes = config.DATOS_GEN / (Path(salida).stem + "_partes")
    partes.mkdir(exist_ok=True)
    tareas = []
    for f in sorted(glob.glob(str(config.DATOS_GEN / nombre / "frag_*.npz"))):
        z = np.load(f)
        ids = sorted({int(k.split("__")[0]) for k in z.files})
        tareas += [(f, i, str(partes / f"{Path(f).stem}__{i}.npz")) for i in ids]
    if max_seq:
        tareas = tareas[:max_seq]
    pendientes = [t for t in tareas if not Path(t[2]).exists() and not Path(t[2] + ".vacia").exists()]
    print(f"{len(tareas)} secuencias, {len(tareas) - len(pendientes)} ya resueltas", flush=True)
    t0 = time.time()
    with Pool(n_proc) as p:
        for k, _ in enumerate(p.imap_unordered(trabajo_a_disco, pendientes, chunksize=2), 1):
            if k % 200 == 0:
                print(k, "/", len(pendientes), round(time.time() - t0), "s", flush=True)
    # unir en el orden de las tareas
    res = []
    for _, _, parte in tareas:
        if Path(parte).exists():
            z = np.load(parte)
            res.append({k: z[k] for k in z.files})
    off, pos = [], 0
    for r in res:
        off.append((pos, len(r["q"])))
        pos += len(r["q"])
    geo_pos = [r["geo_idx"] + p0 for (p0, _), r in zip(off, res)]
    out = {k: np.concatenate([r[k] for r in res], 0) for k in ("q", "qd", "tau", "a", "F", "res", "fase", "R", "Fa", "Fp")}
    out.update(geo_pos=np.concatenate(geo_pos), seq_off=np.array(off), seq_ej=np.array([int(r["ej"]) for r in res]),
               seq_sujeto=np.array([r["sujeto"] for r in res]), seq_semilla=np.array([int(r["semilla"]) for r in res]))
    np.savez(config.DATOS_GEN / salida, **out)
    print("listo", len(res), "secuencias,", pos, "fotogramas,", round(time.time() - t0), "s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=10)
    ap.add_argument("--max_seq", type=int, default=None)
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    main(a.procesos, a.max_seq, a.nombre, a.salida)
