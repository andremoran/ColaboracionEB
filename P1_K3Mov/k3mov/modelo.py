"""K3Mov: codificador-decodificador multimodal del ejercicio, arquitectura Kimi K3 en pequeño.

ENTRADA — matriz de celdas (objeto × modo de darse) por fotograma; cada celda es un escorzo del
mismo movimiento y puede faltar:
    ang_cap  (ángulos, captura)        30 ángulos clínicos + 30 velocidades + máscara por grado
    tras_cap (traslaciones, captura)   pelvis y centro de masa (posición y velocidad, en troncos)
    vid_cap  (video, captura)          17 marcas 2D centradas y normalizadas + visibilidad
    imu_*    (segmento, IMU)           acelerómetro, giroscopio y energía de 4 sitios
    emg      (activación, EMG)         envolventes de hasta 24 músculos + máscara
    forma    (foto, antropometría)     estatura, masa, sexo, IMC (estático)
FUSIÓN — atención sin posición entre celdas (capa equivariante por permutaciones) y promedio
enmascarado (la cosa = componente fija de sus escorzos) + adecuación de la evidencia.
CODIFICADOR — [KDA, KDA, KDA, MLA] × n con Block AttnRes y Stable LatentMoE (SiTU-GLU).
CABEZAS — ejercicio, fase, densidad de repetición (conteo = suma), errores de técnica, calidad,
fatiga, protención (MTP: fotogramas futuros) y reconstrucción de celdas ocultas (cumplimiento).
DECODIFICADOR — genera el movimiento CORRECTO con la misma temporización: 22 rotaciones 6D +
trayectoria de la pelvis + 30 ángulos clínicos.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .capas import AttnResBloque, Bloque, RMSNorm

CELDAS = {  # nombre: (dimensión de entrada, objeto, modo)
    "ang_cap": (90, 0, 0),
    "tras_cap": (12, 1, 0),
    "vid_cap": (51, 2, 0),
    "imu_auricular_I": (7, 3, 1),
    "imu_reloj_D": (7, 3, 1),
    "imu_reloj_I": (7, 3, 1),
    "imu_telefono_D": (7, 3, 1),
    "emg": (48, 4, 2),
    "forma": (4, 5, 3),
}
NOMBRES_CELDAS = list(CELDAS)
N_OBJ, N_MODO = 6, 4
N_EJ = 17          # 16 ejercicios + ninguno
N_FASE = 5
N_ANG = 30
N_ROT = 22


class Tokenizador(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emb = nn.ModuleDict({k: nn.Linear(v[0], d) for k, v in CELDAS.items()})
        self.obj = nn.Embedding(N_OBJ, d)
        self.modo = nn.Embedding(N_MODO, d)
        self.sitio = nn.Embedding(len(CELDAS), d)
        for e in (self.obj, self.modo, self.sitio):      # etiquetas pequeñas: no tapan la señal de la celda
            nn.init.normal_(e.weight, std=0.02)
        self.mezcla = nn.MultiheadAttention(d, 4, batch_first=True)
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.adec = nn.Linear(len(CELDAS) + 1, d)

    def forward(self, celdas, presentes):
        """celdas: dict nombre -> (B,T,dim); presentes: (B,T,C) bool."""
        toks = []
        for i, k in enumerate(NOMBRES_CELDAS):
            o, m = CELDAS[k][1], CELDAS[k][2]
            t = self.emb[k](celdas[k]) + self.obj.weight[o] + self.modo.weight[m] + self.sitio.weight[i]
            toks.append(t)
        X = torch.stack(toks, 2)                                    # (B,T,C,D)
        B, T, C, D = X.shape
        Xf = X.reshape(B * T, C, D)
        pm = presentes.reshape(B * T, C)
        vacio = ~pm.any(-1)
        pm2 = pm.clone()
        pm2[vacio, 0] = True                                        # evita filas vacías en la atención
        h = self.n1(Xf)
        a, _ = self.mezcla(h, h, h, key_padding_mask=~pm2, need_weights=False)
        Xf = Xf + a
        Xf = Xf + self.ff(self.n2(Xf))
        w = pm2.float().unsqueeze(-1)
        fijo = (Xf * w).sum(1) / w.sum(1).clamp(min=1.0)           # componente fija de los escorzos
        adec = torch.cat([pm.float(), pm.float().mean(-1, keepdim=True)], -1)
        return (fijo + self.adec(adec)).view(B, T, D), Xf.view(B, T, C, D)


class Pila(nn.Module):
    """Pila híbrida con Block AttnRes: patrón de mezcladores y tamaño de bloque en capas."""

    def __init__(self, d, patron, cabezas, dk, n_exp, k, tam_bloque=2, ventana=None):
        super().__init__()
        self.capas = nn.ModuleList([Bloque(d, t, cabezas, dk, n_exp, k, ventana) for t in patron])
        self.tam = tam_bloque
        self.final = AttnResBloque(d)
        self.norm = RMSNorm(d)

    def forward(self, x, estados=None, causal=True):
        bloques = [x]
        parcial = torch.zeros_like(x)
        nuevos = []
        for i, capa in enumerate(self.capas):
            est = None if estados is None else estados[i]
            parcial, e = capa(bloques, parcial, est, causal)
            nuevos.append(e)
            if (i + 1) % self.tam == 0:
                bloques = bloques + [parcial]
                parcial = torch.zeros_like(x)
        abiertos = [] if len(self.capas) % self.tam == 0 else [parcial]
        return self.norm(self.final(bloques + abiertos)), nuevos


class K3Mov(nn.Module):
    def __init__(self, d=64, capas=8, cabezas=4, dk=16, n_exp=8, k=2, n_err=60, mtp=4, ventana=None,
                 d_dec=None, capas_dec=4):
        super().__init__()
        patron = (["kda", "kda", "kda", "mla"] * ((capas + 3) // 4))[:capas]
        self.cfg = dict(d=d, capas=capas, cabezas=cabezas, dk=dk, n_exp=n_exp, k=k, n_err=n_err, mtp=mtp,
                        ventana=ventana, d_dec=d_dec or d, capas_dec=capas_dec)
        self.tok = Tokenizador(d)
        self.enc = Pila(d, patron, cabezas, dk, n_exp, k, 2, ventana)
        self.h_ej = nn.Linear(d, N_EJ)
        self.h_fase = nn.Linear(d, N_FASE)
        self.h_rep = nn.Linear(d, 1)
        nn.init.constant_(self.h_rep.bias, -6.0)        # densidad inicial ~0 (pocas repeticiones por ventana)
        self.h_err = nn.Linear(d, n_err)
        self.h_cal = nn.Linear(d, 1)
        self.h_fat = nn.Linear(d, 1)
        self.h_mtp = nn.Linear(d, mtp * N_ANG)
        self.h_rec = nn.ModuleDict({kk: nn.Linear(d, v[0]) for kk, v in CELDAS.items()})
        # decodificador del movimiento correcto
        dd = d_dec or d
        pat_dec = (["kda", "kda", "kda", "mla"] * ((capas_dec + 3) // 4))[:capas_dec]
        self.d_in = nn.Linear(d + N_EJ + CELDAS["ang_cap"][0] + 4, dd)
        self.dec = Pila(dd, pat_dec, cabezas, dk, max(4, n_exp // 2), k, 2, ventana)
        self.h_rot = nn.Linear(dd, N_ROT * 6)
        self.h_raiz = nn.Linear(dd, 3)
        self.h_ang = nn.Linear(dd, N_ANG)

    # ------------------------------------------------------------------------------ uso ---
    def codificar(self, celdas, presentes, estados=None, causal=True):
        x, toks = self.tok(celdas, presentes)
        h, est = self.enc(x, estados, causal)
        return h, toks, est

    def cabezas(self, h):
        B, T, D = h.shape
        return dict(ejercicio=self.h_ej(h), fase=self.h_fase(h), rep=F.softplus(self.h_rep(h)).squeeze(-1),
                    errores=self.h_err(h), calidad=torch.sigmoid(self.h_cal(h)).squeeze(-1),
                    fatiga=torch.sigmoid(self.h_fat(h)).squeeze(-1),
                    mtp=self.h_mtp(h).view(B, T, -1, N_ANG))

    def reconstruir(self, h):
        return {k: m(h) for k, m in self.h_rec.items()}

    def decodificar(self, h, logits_ej, celdas, estados=None, causal=True):
        p = torch.softmax(logits_ej.detach(), -1)
        z = torch.cat([h, p, celdas["ang_cap"], celdas["forma"]], -1)
        g, est = self.dec(self.d_in(z), estados, causal)
        B, T, _ = g.shape
        rot6 = self.h_rot(g).view(B, T, N_ROT, 6)
        ident = torch.tensor([1.0, 0, 0, 0, 1, 0], device=g.device)
        return dict(rot6=rot6 + ident, raiz=self.h_raiz(g), angulos=self.h_ang(g)), est

    def forward(self, celdas, presentes, causal=True, decodificar=True):
        h, toks, _ = self.codificar(celdas, presentes, causal=causal)
        out = self.cabezas(h)
        out["h"] = h
        if decodificar:
            dec, _ = self.decodificar(h, out["ejercicio"], celdas, causal=causal)
            out["dec"] = dec
        return out

    def capas_mla(self):
        for m in self.modules():
            if m.__class__.__name__ == "GatedMLA":
                yield m

    def capas_moe(self):
        for m in self.modules():
            if m.__class__.__name__ == "StableLatentMoE":
                yield m

    def n_parametros(self):
        return sum(p.numel() for p in self.parameters())

    def n_activos(self):
        """Parámetros activos por fotograma (en el MoE solo k de n expertos trabajan)."""
        tot = self.n_parametros()
        for m in self.capas_moe():
            per = (m.Wg[0].numel() + m.Wu[0].numel() + m.Wo[0].numel())
            tot -= per * (m.n - m.k)
        return tot
