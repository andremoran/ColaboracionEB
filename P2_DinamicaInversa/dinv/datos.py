"""Datos del Proyecto 2: de la captura ruidosa (como la de index3) + masa, estatura y carga, a los
momentos articulares, reacciones del suelo y potencias calculados con la física sobre la cinemática
LIMPIA del generador (verdad de terreno).

Entradas por fotograma (como las tendría Version49):
  - 33 marcas «world» (métricas, centradas en la cadera, con ruido de profundidad) / estatura
  - trayectoria de la pelvis en la imagen (normalizada por el tronco en la imagen)
  - masa, estatura, sexo, carga en las manos (constantes)
Salidas: 24 momentos anatómicos (N·m/kg), reacción de cada pie (fracción del peso), 22 potencias (W/kg).
"""
import glob

import numpy as np
import torch

from nucleo import config
from nucleo.ejercicios import NOMBRES_EJ

CAMPOS = ["mundo_captura", "captura_T", "momentos", "grf", "contacto", "potencia", "G6", "P", "L", "fase"]


def _apoyos_pies(ej):
    from nucleo.ejercicios import EJERCICIOS
    return sorted(EJERCICIOS[NOMBRES_EJ[ej]]["apoyos"])


def construir_cache(nombre=None):
    """Caché en disco (una carpeta de .npy abiertos con memmap), construida en dos pasadas para no
    reunir los 3.2 millones de fotogramas en memoria: la primera mide, la segunda escribe."""
    nombre = nombre or config.CONJUNTO_DIN
    dst = config.DATOS_GEN / f"{nombre}_p2_mm"
    if (dst / "listo.txt").exists():
        return dst
    if not (config.DATOS_GEN / nombre / "info.json").exists():
        raise RuntimeError(f"El conjunto {nombre} no está completo (falta info.json): no se construye una caché parcial")
    frags = sorted(glob.glob(str(config.DATOS_GEN / nombre / "frag_*.npz")))
    # 1.ª pasada: secuencias válidas, largos y formas
    lista, formas = [], {}
    for f in frags:
        z = np.load(f)
        for i in sorted({int(k.split("__")[0]) for k in z.files}):
            g = lambda k: z[f"{i}__{k}"]
            if any(np.isinf(g(k).astype(np.float32)).any() for k in ("mundo_captura", "momentos", "potencia", "grf")):
                continue
            lista.append((f, i, int(g("T"))))
            if not formas:
                for k in CAMPOS:
                    if k == "grf":
                        formas[k] = ((2, 3), np.float16)
                    elif k == "contacto":
                        formas[k] = ((2,), np.int8)
                    else:
                        x = g(k)
                        formas[k] = (x.shape[1:], x.dtype)
    N = sum(t for _, _, t in lista)
    dst.mkdir(exist_ok=True)
    mm = {k: np.lib.format.open_memmap(dst / f"{k}.npy", mode="w+", dtype=dt, shape=(N,) + sh)
          for k, (sh, dt) in formas.items()}
    seq = dict(inicio=[], T=[], ej=[], sujeto=[], carga=[], apoyos=[])
    pos = 0
    z, f_act = None, None
    # 2.ª pasada: escribe cada secuencia en su lugar
    for f, i, T in lista:
        if f != f_act:
            z, f_act = np.load(f), f
        g = lambda k: z[f"{i}__{k}"]
        ej = int(g("ejercicio"))
        apoyos = _apoyos_pies(ej)
        sl = slice(pos, pos + T)
        for k in CAMPOS:
            if k == "grf":
                grf = g("grf").astype(np.float32)
                gp = np.full((T, 2, 3), np.nan, np.float32)      # [pie_D, pie_I]; NaN si no apoya los pies
                for kk, a in enumerate(apoyos):
                    if a in ("pie_D", "pie_I"):
                        gp[:, 0 if a == "pie_D" else 1] = grf[:, kk]
                mm[k][sl] = gp.astype(np.float16)
            elif k == "contacto":
                c = np.zeros((T, 2), np.int8)
                cc = g("contacto")
                for kk, a in enumerate(apoyos):
                    if a in ("pie_D", "pie_I"):
                        c[:, 0 if a == "pie_D" else 1] = cc[:, kk]
                mm[k][sl] = c
            else:
                mm[k][sl] = g(k)
        seq["inicio"].append(pos); seq["T"].append(T); seq["ej"].append(ej)
        seq["sujeto"].append(g("sujeto")); seq["carga"].append(float(g("carga"))); seq["apoyos"].append(",".join(apoyos))
        pos += T
    for v in mm.values():
        v.flush()
    del mm
    for k, v in dict(seq_inicio=np.array(seq["inicio"]), seq_T=np.array(seq["T"]), seq_ej=np.array(seq["ej"]),
                     seq_sujeto=np.array(seq["sujeto"]), seq_carga=np.array(seq["carga"]),
                     seq_apoyos=np.array(seq["apoyos"])).items():
        np.save(dst / f"{k}.npy", v)
    (dst / "listo.txt").write_text("\n".join(CAMPOS + ["seq_inicio", "seq_T", "seq_ej", "seq_sujeto", "seq_carga",
                                                       "seq_apoyos"]), encoding="utf-8")
    return dst


def abrir_cache(nombre=None):
    """Diccionario de arreglos en memmap (los comparten todos los procesos)."""
    dst = construir_cache(nombre)
    claves = (dst / "listo.txt").read_text(encoding="utf-8").split("\n")
    return {k: np.load(dst / f"{k}.npy", mmap_mode="r") for k in claves}


def entradas(W, capT, sujeto, carga, fps=30.0):
    """W (T,33,3) marcas world centradas en la cadera (m); capT (T,33,3) espacio T de la imagen.
    Devuelve (T, F) float32."""
    est, masa, sexo = float(sujeto[0]), float(sujeto[1]), float(sujeto[2])
    Wn = np.nan_to_num(W / est)
    v = np.gradient(Wn, 1 / fps, axis=0)
    mh = np.nanmean(capT[:, [23, 24]], 1)
    ms = np.nanmean(capT[:, [11, 12]], 1)
    tor = np.nanmedian(np.linalg.norm(ms - mh, axis=-1)) + 1e-6
    p = np.nan_to_num((mh - np.nanmedian(mh, 0)) / tor)
    pv = np.gradient(p, 1 / fps, axis=0)
    pa = np.gradient(pv, 1 / fps, axis=0)
    const = np.array([(est - 1.7) / 0.1, (masa - 70) / 15, sexo, carga / 10.0])
    X = np.concatenate([Wn.reshape(len(W), -1), np.clip(v.reshape(len(W), -1), -10, 10), p, pv, np.clip(pa, -50, 50) / 10,
                        np.repeat(const[None], len(W), 0)], -1)
    return X.astype(np.float32)


class Ventanas(torch.utils.data.Dataset):
    def __init__(self, z, idx, T_win=96, n=4000, semilla=0, ejercicios=None):
        self.z, self.T, self.n = z, T_win, n
        self.idx = [s for s in idx if ejercicios is None or NOMBRES_EJ[int(z["seq_ej"][s])] in ejercicios]
        self.semilla = semilla

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        z = self.z
        rng = np.random.default_rng([self.semilla, int(i)])      # ventana fija por índice (validación estable)
        s = int(self.idx[rng.integers(len(self.idx))])
        i0, T = int(z["seq_inicio"][s]), int(z["seq_T"][s])
        a = int(rng.integers(0, max(1, T - self.T + 1)))
        b = min(T, a + self.T)
        sl = slice(i0 + a, i0 + b)
        suj = z["seq_sujeto"][s]
        masa = float(suj[1])
        X = entradas(z["mundo_captura"][sl].astype(np.float64), z["captura_T"][sl].astype(np.float64), suj,
                     float(z["seq_carga"][s]))
        M = z["momentos"][sl].astype(np.float32) / masa                              # N·m/kg
        F = z["grf"][sl].astype(np.float32) / (masa * 9.81)                          # fracción del peso
        mF = np.isfinite(F).all(-1).astype(np.float32)
        F = np.nan_to_num(F)
        Pw = z["potencia"][sl].astype(np.float32) / masa                             # W/kg
        return dict(X=torch.from_numpy(X), M=torch.from_numpy(np.nan_to_num(M)), F=torch.from_numpy(F),
                    mF=torch.from_numpy(mF), Pw=torch.from_numpy(np.nan_to_num(Pw)),
                    G6=torch.from_numpy(z["G6"][sl].astype(np.float32)), P=torch.from_numpy(z["P"][sl].astype(np.float32)),
                    contacto=torch.from_numpy(z["contacto"][sl].astype(np.float32)),
                    masa=torch.tensor(masa), estatura=torch.tensor(float(suj[0])), carga=torch.tensor(float(z["seq_carga"][s])),
                    largo=torch.tensor(b - a))


def colar(lote):
    Tm = max(int(x["largo"]) for x in lote)
    out = {}
    for k in lote[0]:
        if lote[0][k].ndim == 0:
            out[k] = torch.stack([x[k] for x in lote])
        else:
            out[k] = torch.stack([torch.cat([x[k], x[k].new_zeros((Tm - x[k].shape[0],) + tuple(x[k].shape[1:]))]) for x in lote])
    out["valido"] = torch.arange(Tm)[None] < out["largo"][:, None]
    return out
