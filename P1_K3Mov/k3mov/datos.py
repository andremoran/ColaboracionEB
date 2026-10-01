"""Datos de K3Mov: caché consolidada del conjunto sintético, construcción de celdas por fotograma
(la misma para el sintético, MM-Fit, EC3D y las exportaciones de index3) y aumentos.
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch

from nucleo import config
from nucleo.angulos_clinicos import CLAVES
from nucleo.antropometria import com_index3
from nucleo.ejercicios import CATALOGO_ERRORES, NOMBRES_EJ
from .modelo import CELDAS, NOMBRES_CELDAS

PUNTOS_2D = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
N_ERR = len(CATALOGO_ERRORES)
NINGUNO = len(NOMBRES_EJ)            # clase 16
SITIOS = ["auricular_I", "reloj_D", "reloj_I", "telefono_D"]

CAMPOS = ["angulos_captura", "captura_T", "captura_vis", "imu", "activacion", "fase", "rep", "fatiga", "angulos",
          "R6_corr", "raiz6_corr", "raiz_t_corr", "angulos_corr", "L_corr", "err"]


# ================================================================ caché ===
def _valida(g):
    """Secuencias con desbordes numéricos (float16) o captura degenerada se descartan."""
    malas = [k for k in ("captura_T", "angulos_captura", "L_corr", "R6_corr", "imu", "activacion")
             if np.isinf(g(k).astype(np.float32)).any()]
    return not malas and np.isnan(g("captura_T").astype(np.float32)).mean() <= 0.05


def construir_cache(nombre=None):
    """Caché consolidada de K3Mov en disco: una carpeta de .npy (abiertos luego con memmap), escrita en
    dos pasadas (la primera mide, la segunda escribe) para no reunir millones de fotogramas en memoria.
    Si existe una caché antigua en .npz se reutiliza."""
    nombre = nombre or config.CONJUNTO
    antigua = config.DATOS_GEN / f"{nombre}_p1.npz"
    dst = config.DATOS_GEN / f"{nombre}_p1_mm"
    if (dst / "listo.txt").exists():
        return dst
    if antigua.exists():
        return antigua
    if not (config.DATOS_GEN / nombre / "info.json").exists():
        raise RuntimeError(f"El conjunto {nombre} no está completo (falta info.json): no se construye una caché parcial")
    frags = sorted(glob.glob(str(config.DATOS_GEN / nombre / "frag_*.npz")))
    lista, formas = [], {}
    for f in frags:                                     # 1.ª pasada: qué secuencias entran y cuánto miden
        z = np.load(f)
        for i in sorted({int(k.split("__")[0]) for k in z.files}):
            g = lambda k: z[f"{i}__{k}"]
            if not _valida(g):
                continue
            lista.append((f, i, int(g("T"))))
            if not formas:
                for k in CAMPOS:
                    x = g(k)
                    formas[k] = ((8,), np.float16) if k == "err" else (x.shape[1:], x.dtype)
    N = sum(t for _, _, t in lista)
    dst.mkdir(exist_ok=True)
    mm = {k: np.lib.format.open_memmap(dst / f"{k}.npy", mode="w+", dtype=dt, shape=(N,) + sh)
          for k, (sh, dt) in formas.items()}
    seq = dict(inicio=[], T=[], ejercicio=[], sujeto=[], err_idx=[], eventos=[], semilla=[], camara=[], carga=[])
    pos, z, f_act = 0, None, None
    for f, i, T in lista:                               # 2.ª pasada: cada secuencia en su lugar
        if f != f_act:
            z, f_act = np.load(f), f
        g = lambda k: z[f"{i}__{k}"]
        sl = slice(pos, pos + T)
        for k in CAMPOS:
            if k == "err":
                e = np.zeros((T, 8), np.float16)
                v = g("err")
                e[:, :v.shape[1]] = v
                mm[k][sl] = e
            else:
                mm[k][sl] = g(k)
        ei = np.full(8, -1, np.int16)
        v = g("err_idx")
        ei[:len(v)] = v
        seq["inicio"].append(pos); seq["T"].append(T); seq["ejercicio"].append(int(g("ejercicio")))
        seq["sujeto"].append(g("sujeto")); seq["err_idx"].append(ei); seq["semilla"].append(int(g("semilla")))
        seq["eventos"].append(g("eventos")); seq["camara"].append(g("camara")); seq["carga"].append(float(g("carga")))
        pos += T
    for v in mm.values():
        v.flush()
    del mm
    ev_all = np.concatenate([np.c_[np.full(len(e), i), e] for i, e in enumerate(seq["eventos"])], 0)
    extra = dict(seq_inicio=np.array(seq["inicio"]), seq_T=np.array(seq["T"]), seq_ej=np.array(seq["ejercicio"]),
                 seq_sujeto=np.array(seq["sujeto"]), seq_err_idx=np.array(seq["err_idx"]),
                 seq_semilla=np.array(seq["semilla"]), seq_camara=np.array(seq["camara"]),
                 seq_carga=np.array(seq["carga"]), eventos=ev_all)
    for k, v in extra.items():
        np.save(dst / f"{k}.npy", v)
    (dst / "listo.txt").write_text("\n".join(CAMPOS + list(extra)), encoding="utf-8")
    return dst


def _a_memmap(ruta):
    """Carpeta de .npy de la caché (abiertos con memmap: los procesos del cargador de datos comparten
    las mismas páginas en vez de copiar 3.4 GB cada uno). Una caché antigua en .npz se desempaqueta una vez."""
    ruta = Path(ruta)
    if ruta.is_dir():
        return ruta, (ruta / "listo.txt").read_text(encoding="utf-8").split("\n")
    carpeta = ruta.with_name(ruta.stem + "_mm")
    listo = carpeta / "listo.txt"
    if not listo.exists():
        carpeta.mkdir(exist_ok=True)
        z = np.load(ruta)
        for k in z.files:
            np.save(carpeta / f"{k}.npy", z[k])
        listo.write_text("\n".join(z.files), encoding="utf-8")
    return carpeta, listo.read_text(encoding="utf-8").split("\n")


class Cache:
    def __init__(self, ruta):
        self.ruta = str(ruta)
        carpeta, claves = _a_memmap(ruta)
        self.d = {k: np.load(carpeta / f"{k}.npy", mmap_mode="r") for k in claves}
        self.n = len(self.d["seq_T"])
        ev = np.asarray(self.d["eventos"])
        self.ev_por_seq = [ev[ev[:, 0] == i] for i in range(self.n)]

    def __getstate__(self):                # al pasar a otro proceso solo viaja la ruta
        return {"ruta": self.ruta}

    def __setstate__(self, estado):
        self.__init__(estado["ruta"])

    def particion(self, frac_val=0.08, frac_test=0.10, semilla=0):
        rng = np.random.default_rng(semilla)
        idx = rng.permutation(self.n)
        nv, nt = int(self.n * frac_val), int(self.n * frac_test)
        return idx[nt + nv:], idx[nt:nt + nv], idx[:nt]


# ====================================================== celdas (features) ===
def _vel(x, fps):
    v = np.gradient(np.nan_to_num(x), 1.0 / fps, axis=0)
    return np.where(np.isfinite(x), v, 0.0)


def celda_angulos(A, fps):
    m = np.isfinite(A)
    a = np.where(m, A, 0.0) / 90.0
    v = np.clip(_vel(A, fps) / 300.0, -3, 3)
    return np.concatenate([a, v, m.astype(np.float32)], -1), m.any(-1)


def _torso(L):
    ms = np.nanmean(L[:, [11, 12]], 1)
    mh = np.nanmean(L[:, [23, 24]], 1)
    t = np.nanmedian(np.linalg.norm(ms - mh, axis=-1))
    return mh, (t if np.isfinite(t) and t > 1e-6 else 1.0)


def celda_traslacion(L, fps):
    """Pelvis y centro de masa (posición respecto a la mediana y velocidad), en longitudes de tronco."""
    mh, tor = _torso(L)
    ref = np.nanmedian(mh, 0)
    p = (mh - ref) / tor
    Lc = np.where(np.isfinite(L), L, np.nan)
    Lf = Lc.copy()
    for j in range(33):                               # rellena puntos faltantes con la pelvis
        falta = ~np.isfinite(Lf[:, j, 0])
        Lf[falta, j] = mh[falta]
    c = (com_index3(Lf) - mh) / tor
    x = np.concatenate([p, _vel(p, fps), c, _vel(c, fps)], -1)
    x = np.clip(np.nan_to_num(x), -5, 5)
    return x, np.isfinite(mh).all(-1)


def celda_video(L, vis=None):
    """17 marcas 2D (plano de la imagen = x,y del espacio T) centradas en la pelvis, en troncos."""
    mh, tor = _torso(L)
    P = (L[:, PUNTOS_2D, :2] - mh[:, None, :2]) / tor
    v = np.ones((L.shape[0], len(PUNTOS_2D))) if vis is None else vis[:, PUNTOS_2D]
    ok = np.isfinite(P).all(-1)
    v = np.where(ok, v, 0.0)
    P = np.where(ok[..., None], P, 0.0)
    return np.concatenate([P.reshape(len(P), -1), v], -1), ok.sum(-1) >= 8


def celda_imu(imu):
    """imu (T, 4, 7): acc m/s², giro rad/s, energía -> escalado."""
    x = np.concatenate([imu[..., :3] / 9.81, imu[..., 3:6] / 5.0, imu[..., 6:7] / 5.0], -1)
    pres = np.abs(imu).sum(-1) > 0
    return x, pres


def celda_emg(act, presentes=None):
    T = act.shape[0]
    m = np.ones((T, act.shape[1])) if presentes is None else np.broadcast_to(presentes, (T, act.shape[1]))
    return np.concatenate([np.where(m > 0, act, 0.0), m], -1), m.any(-1)


def celda_forma(sujeto, T):
    if sujeto is None:
        return np.zeros((T, 4)), np.zeros(T, bool)
    est, masa, sexo = sujeto
    imc = masa / est ** 2
    x = np.array([(est - 1.7) / 0.1, (masa - 70) / 15, sexo, (imc - 24) / 4])
    return np.repeat(x[None], T, 0), np.ones(T, bool)


def armar_celdas(A=None, L=None, vis=None, imu=None, act=None, act_mask=None, sujeto=None, fps=30.0, T=None):
    """Devuelve (celdas dict nombre -> (T,dim) float32, presentes (T,C) bool)."""
    T = T or next(x.shape[0] for x in (A, L, imu, act) if x is not None)
    cel, pres = {}, {}
    for k, (dim, _, _) in CELDAS.items():
        cel[k] = np.zeros((T, dim), np.float32)
        pres[k] = np.zeros(T, bool)
    if A is not None:
        cel["ang_cap"], pres["ang_cap"] = celda_angulos(A, fps)
    if L is not None:
        cel["tras_cap"], pres["tras_cap"] = celda_traslacion(L, fps)
        cel["vid_cap"], pres["vid_cap"] = celda_video(L, vis)
    if imu is not None:
        x, p = celda_imu(imu)
        for i, s in enumerate(SITIOS):
            cel["imu_" + s], pres["imu_" + s] = x[:, i], p[:, i]
    if act is not None:
        cel["emg"], pres["emg"] = celda_emg(act, act_mask)
    cel["forma"], pres["forma"] = celda_forma(sujeto, T)
    P = np.stack([pres[k] for k in NOMBRES_CELDAS], -1)
    return {k: v.astype(np.float32) for k, v in cel.items()}, P


# ============================================================== aumentos ===
def ema_index3(A, alfa):
    """Suavizado exponencial por grado de libertad como index3 (clin = p + (x - p)·α)."""
    B = A.copy()
    prev = None
    for t in range(len(A)):
        x = A[t]
        if prev is not None:
            ok = np.isfinite(x) & np.isfinite(prev)
            x = np.where(ok, prev + (x - prev) * alfa, x)
        B[t] = x
        prev = np.where(np.isfinite(x), x, prev) if prev is not None else x
    return B


def rot_aleatoria(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def aumentar_imu(imu, rng):
    imu = imu.copy()
    for i in range(imu.shape[1]):
        R = rot_aleatoria(rng)
        imu[:, i, :3] = imu[:, i, :3] @ R.T
        imu[:, i, 3:6] = imu[:, i, 3:6] @ R.T
        imu[:, i, :6] *= rng.normal(1, 0.03)
    return imu


# ============================================================ muestreo ===
class Ventanas(torch.utils.data.Dataset):
    """Ventanas aleatorias del conjunto sintético con todas las etiquetas."""

    def __init__(self, cache, idx_seq, T_win=192, n_por_epoca=6000, semilla=0, aumentar=True, p_celda=0.35):
        self.c, self.idx, self.T, self.n = cache, np.asarray(idx_seq), T_win, n_por_epoca
        self.semilla = semilla
        self.aum, self.p = aumentar, p_celda

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # generador propio de cada índice: los procesos del cargador no repiten las mismas ventanas
        rng = np.random.default_rng([self.semilla, int(i)])
        d = self.c.d
        s = int(self.idx[rng.integers(len(self.idx))])
        i0, T = int(d["seq_inicio"][s]), int(d["seq_T"][s])
        W = self.T
        a = int(rng.integers(0, max(1, T - W + 1)))
        b = min(T, a + W)
        sl = slice(i0 + a, i0 + b)
        A = d["angulos_captura"][sl].astype(np.float64)
        capT = d["captura_T"][sl].astype(np.float64)
        vis = d["captura_vis"][sl].astype(np.float64)
        imu = d["imu"][sl].astype(np.float64)
        act = d["activacion"][sl].astype(np.float64)
        suj = d["seq_sujeto"][s]
        act_mask = None
        if self.aum:
            if rng.random() < 0.6:
                A = ema_index3(A, rng.uniform(0.4, 1.0))
            imu = aumentar_imu(imu, rng)
            k = int(rng.integers(0, 9))
            act_mask = np.zeros(act.shape[1])
            if k > 0:
                act_mask[rng.choice(act.shape[1], size=k, replace=False)] = 1
            # grados de libertad ausentes (como MM-Fit sin manos/pies, o index3 sin cara)
            if rng.random() < 0.3:
                A[:, [CLAVES.index(x) for x in CLAVES if x.startswith(("wrist", "elbowR_pron", "elbowL_pron", "neck"))]] = np.nan
            if rng.random() < 0.2:
                A[:, [CLAVES.index(x) for x in CLAVES if x.startswith("ankle")]] = np.nan
        cel, P = armar_celdas(A=A, L=capT, vis=vis, imu=imu, act=act, act_mask=act_mask,
                              sujeto=suj if rng.random() < 0.8 or not self.aum else None, T=b - a)
        if self.aum:                    # celdas completas ausentes (horizonte interno)
            quita = rng.random(P.shape[1]) < self.p
            if quita.all() or not (P[:, ~quita].any()):
                quita[0] = False
            P[:, quita] = False
        oculta = np.zeros_like(P)       # celdas presentes que se ocultan para reconstruirlas
        if self.aum:
            oc = rng.random(P.shape[1]) < 0.15
            oculta[:, oc] = P[:, oc]
        # etiquetas
        fase = d["fase"][sl].astype(np.int64)
        ej = np.where(fase > 0, int(d["seq_ej"][s]), NINGUNO)
        ev = self.c.ev_por_seq[s]
        rep_dens = np.zeros(b - a)
        fps = 30.0
        for e in ev:
            t_fin = int(round(e[6] * fps)) - a               # e: [seq, rep, inicio, m1a, m1b, p1b, m2b, fin, tcon, texc]
            if 0 <= t_fin < b - a:
                g = np.exp(-0.5 * ((np.arange(b - a) - t_fin) / 2.0) ** 2)
                rep_dens += g / g.sum()
        err = np.zeros((b - a, N_ERR), np.float32)
        ei = d["seq_err_idx"][s]
        e_arr = d["err"][sl].astype(np.float32)
        for j, gidx in enumerate(ei):
            if gidx >= 0:
                err[:, gidx] = e_arr[:, j]
        mask_err = np.zeros(N_ERR, np.float32)
        for gidx in ei:
            if gidx >= 0:
                mask_err[gidx] = 1.0
        cal = 1.0 - err.max(-1)
        Ac = d["angulos"][sl].astype(np.float64)
        mtp = np.zeros((b - a, 4, 30))
        mtp_m = np.zeros((b - a, 4, 30))
        for k in range(1, 5):
            if b - a > k:
                dlt = (Ac[k:] - Ac[:-k]) / 90.0
                mtp[:-k, k - 1] = np.nan_to_num(dlt)
                mtp_m[:-k, k - 1] = np.isfinite(dlt)
        raiz = d["raiz_t_corr"][sl].astype(np.float64)
        est = float(suj[0])
        raiz = (raiz - raiz[0]) / est
        out = dict(
            celdas={k: torch.from_numpy(v) for k, v in cel.items()}, presentes=torch.from_numpy(P),
            oculta=torch.from_numpy(oculta), ejercicio=torch.from_numpy(ej), fase=torch.from_numpy(fase),
            rep=torch.from_numpy(rep_dens.astype(np.float32)), errores=torch.from_numpy(err),
            mask_err=torch.from_numpy(mask_err), calidad=torch.from_numpy(cal.astype(np.float32)),
            fatiga=torch.from_numpy(d["fatiga"][sl].astype(np.float32)),
            mtp=torch.from_numpy(mtp.astype(np.float32)), mtp_m=torch.from_numpy(mtp_m.astype(np.float32)),
            rot6=torch.from_numpy(_rot_objetivo(d["R6_corr"][sl], d["raiz6_corr"][sl])), raiz=torch.from_numpy(raiz.astype(np.float32)),
            ang_corr=torch.from_numpy(np.nan_to_num(d["angulos_corr"][sl].astype(np.float32) / 90.0)),
            ang_corr_m=torch.from_numpy(np.isfinite(d["angulos_corr"][sl].astype(np.float32)).astype(np.float32)),
            L_corr=torch.from_numpy(d["L_corr"][sl].astype(np.float32)), estatura=torch.tensor(est, dtype=torch.float32),
            largo=torch.tensor(b - a))
        return out


def _rot_objetivo(R6, raiz6):
    """La articulación 0 lleva la orientación global de la pelvis (la local es la identidad)."""
    R = R6.astype(np.float32).copy()
    R[:, 0] = raiz6.astype(np.float32)
    return R


def colar(lote):
    """Rellena a la longitud máxima del lote y crea la máscara de validez temporal."""
    Tm = max(int(x["largo"]) for x in lote)
    def pad(t, T):
        if t.shape[0] == T:
            return t
        z = torch.zeros((T - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)
        return torch.cat([t, z], 0)
    out = {}
    for k in lote[0]:
        if k == "celdas":
            out[k] = {c: torch.stack([pad(x[k][c], Tm) for x in lote]) for c in lote[0][k]}
        elif k in ("mask_err", "estatura", "largo"):
            out[k] = torch.stack([x[k] for x in lote])
        else:
            out[k] = torch.stack([pad(x[k], Tm) for x in lote])
    out["valido"] = torch.arange(Tm)[None] < out["largo"][:, None]
    return out
