"""Capas de la arquitectura Kimi K3 llevadas a escala de movimiento humano.

Ecuaciones tomadas del reporte técnico de Kimi K3 (Kimi Team, 2026) y de sus antecedentes:
- KDA (Kimi Delta Attention): S_t = (I - β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ,  õ_t = S_tᵀ q_t,
  con decaimiento por canal acotado g = g_min·σ(e^{A}·z), α = exp(g), g_min = -5, y salida
  y = W_o[σ(W_g x) ⊙ RMSNorm(õ)].  Es recurrente: en vivo cuesta O(1) por fotograma.
- MLA con compuerta: atención completa con clave/valor comprimidos en un latente c = W_dkv x, sin
  codificación de posición (NoPE): el orden lo aporta la recurrencia de KDA. y = W_o[σ(W_g x) ⊙ õ].
- Block AttnRes: la entrada de cada subcapa es una atención sobre la profundidad,
  h = Σ_i softmax_i(w_l · RMSNorm(V_i)) V_i, con V = [embebido, bloques completos, bloque parcial].
- Stable LatentMoE: y = Σ_j E_j^shared(x) + W↑ RMSNorm(Σ_{i∈TopK} p_i E_i(W↓ x)), puntuación
  σ(W_r x) con sesgo solo para elegir, balanceo por cuantiles (Quantile Balancing).
- SiTU-GLU(x) = [β1 tanh(W_g x/β1) ⊙ σ(W_g x)] ⊙ [β2 tanh(W_u x/β2)], β1 = 4, β2 = 25.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6, afin=True):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d)) if afin else None

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * self.w if self.w is not None else y


class SiTUGLU(nn.Module):
    """Unidad SiTU-GLU con topes suaves: |salida intermedia| <= β1·β2 = 100."""

    def __init__(self, d_in, d_hid, d_out=None, b1=4.0, b2=25.0):
        super().__init__()
        self.g = nn.Linear(d_in, d_hid, bias=False)
        self.u = nn.Linear(d_in, d_hid, bias=False)
        self.o = nn.Linear(d_hid, d_out or d_in, bias=False)
        self.b1, self.b2 = b1, b2

    def forward(self, x):
        g = self.g(x)
        a = self.b1 * torch.tanh(g / self.b1) * torch.sigmoid(g)
        u = self.b2 * torch.tanh(self.u(x) / self.b2)
        return self.o(a * u)


class ConvCorta(nn.Module):
    """Convolución causal en profundidad (núcleo 4) con SiLU, como en KDA."""

    def __init__(self, d, k=4):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(d, k) * (1.0 / math.sqrt(k)))

    def forward(self, x, estado=None):
        # x (B,T,D); estado (B,k-1,D) con las últimas entradas (modo en vivo)
        B, T, D = x.shape
        if estado is None:
            estado = x.new_zeros(B, self.k - 1, D)
        xx = torch.cat([estado, x], 1)                         # (B, T+k-1, D)
        y = sum(xx[:, i:i + T] * self.w[:, i] for i in range(self.k))
        return F.silu(y), xx[:, -(self.k - 1):]


def barrido_delta(q, k, v, beta, alfa, S0):
    """Regla delta con decaimiento por canal resuelta por prefijo paralelo (Hillis–Steele).

    S_t = A_t S_{t-1} + b_t con A_t = (I - β k kᵀ) Diag(α) = Diag(α) - β k (k ⊙ α)ᵀ y b_t = β k vᵀ.
    La composición (A2,b2)∘(A1,b1) = (A2 A1, A2 b1 + b2) es asociativa: log2(T) pasos vectorizados.
    q,k (B,T,H,dk) · v (B,T,H,dv) · beta (B,T,H,1) · alfa (B,T,H,dk) · S0 (B,H,dk,dv)."""
    B, T, H, dk = k.shape
    kb = k * beta
    A = torch.diag_embed(alfa) - kb.unsqueeze(-1) * (k * alfa).unsqueeze(-2)     # (B,T,H,dk,dk)
    Bm = kb.unsqueeze(-1) * v.unsqueeze(-2)                                    # (B,T,H,dk,dv)
    A = A.permute(0, 2, 1, 3, 4)                                               # (B,H,T,dk,dk)
    Bm = Bm.permute(0, 2, 1, 3, 4)
    d = 1
    while d < T:
        A_prev, B_prev = A[:, :, :-d], Bm[:, :, :-d]
        A_cur, B_cur = A[:, :, d:], Bm[:, :, d:]
        Bm = torch.cat([Bm[:, :, :d], A_cur @ B_prev + B_cur], 2)
        A = torch.cat([A[:, :, :d], A_cur @ A_prev], 2)
        d *= 2
    S = A @ S0.unsqueeze(2) + Bm                                               # (B,H,T,dk,dv)
    o = torch.einsum("bhtk,bhtkv->bthv", q.permute(0, 2, 1, 3), S)
    return o, S[:, :, -1]


class KDA(nn.Module):
    """Atención lineal con regla delta y compuertas finas (una capa)."""

    def __init__(self, d, cabezas=4, dk=16, dv=16, g_min=-5.0):
        super().__init__()
        self.h, self.dk, self.dv, self.g_min = cabezas, dk, dv, g_min
        self.q = nn.Linear(d, cabezas * dk, bias=False)
        self.k = nn.Linear(d, cabezas * dk, bias=False)
        self.v = nn.Linear(d, cabezas * dv, bias=False)
        self.cq, self.ck, self.cv = ConvCorta(cabezas * dk), ConvCorta(cabezas * dk), ConvCorta(cabezas * dv)
        self.beta = nn.Linear(d, cabezas, bias=True)
        self.z = nn.Sequential(nn.Linear(d, dk, bias=False), nn.Linear(dk, cabezas * dk, bias=True))  # bajo rango
        self.A = nn.Parameter(torch.zeros(cabezas, dk))
        # memorias escalonadas: σ(z) inicial en [e^-7.5, 0.3] -> α = exp(g_min·σ(z)) de ~0.22 a ~0.997
        # (constantes de tiempo de 1 fotograma a ~10 s a 30 fps), como la inicialización de Kimi Linear/Mamba
        with torch.no_grad():
            self.z[1].bias.copy_(torch.linspace(-7.5, -0.85, cabezas * dk)[torch.randperm(cabezas * dk)])
        self.g = nn.Linear(d, cabezas * dv, bias=False)
        self.norm = RMSNorm(dv)
        self.o = nn.Linear(cabezas * dv, d, bias=False)
        self.barrido = True

    def forward(self, x, estado=None):
        B, T, _ = x.shape
        H, dk, dv = self.h, self.dk, self.dv
        est = estado or {}
        q, eq = self.cq(self.q(x), est.get("cq"))
        k, ek = self.ck(self.k(x), est.get("ck"))
        v, ev = self.cv(self.v(x), est.get("cv"))
        q = F.normalize(q.view(B, T, H, dk), dim=-1)
        k = F.normalize(k.view(B, T, H, dk), dim=-1)
        v = v.view(B, T, H, dv)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        z = self.z(x).view(B, T, H, dk)
        g = self.g_min * torch.sigmoid(torch.exp(self.A) * z)       # (g_min, 0)
        alfa = torch.exp(g)                                         # (e^g_min, 1)
        S = est.get("S")
        if S is None:
            S = x.new_zeros(B, H, dk, dv)
        if T == 1 or not self.barrido:                              # en vivo: recurrencia O(1)
            salidas = []
            for t in range(T):
                S = S * alfa[:, t].unsqueeze(-1)                    # Diag(α) S
                kt, vt, bt = k[:, t], v[:, t], beta[:, t]
                pred = torch.einsum("bhk,bhkv->bhv", kt, S)         # Sᵀ k
                S = S + torch.einsum("bhk,bhv->bhkv", kt * bt, vt - pred)   # regla delta
                salidas.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
            o = torch.stack(salidas, 1)                             # (B,T,H,dv)
        else:
            o, S = barrido_delta(q, k, v, beta, alfa, S)
        o = self.norm(o) * torch.sigmoid(self.g(x)).view(B, T, H, dv)
        y = self.o(o.reshape(B, T, H * dv))
        return y, {"S": S, "cq": eq, "ck": ek, "cv": ev}


class GatedMLA(nn.Module):
    """Atención completa con latente de clave/valor, sin posición, con compuerta de salida.
    `ventana` limita el contexto (en vivo); `causal` para el modo en vivo."""

    def __init__(self, d, cabezas=4, dh=16, d_lat=32, ventana=None):
        super().__init__()
        self.h, self.dh, self.ventana = cabezas, dh, ventana
        self.q = nn.Linear(d, cabezas * dh, bias=False)
        self.dkv = nn.Linear(d, d_lat, bias=False)
        self.uk = nn.Linear(d_lat, cabezas * dh, bias=False)
        self.uv = nn.Linear(d_lat, cabezas * dh, bias=False)
        self.nkv = RMSNorm(d_lat)
        self.g = nn.Linear(d, cabezas * dh, bias=False)
        self.o = nn.Linear(cabezas * dh, d, bias=False)
        self.max_logit = torch.zeros(cabezas)                       # para el recorte QK

    def forward(self, x, estado=None, causal=True):
        B, T, _ = x.shape
        H, dh = self.h, self.dh
        c = self.nkv(self.dkv(x))                                   # (B,T,d_lat): caché barata
        if estado is not None and estado.get("c") is not None:
            c_all = torch.cat([estado["c"], c], 1)
        else:
            c_all = c
        if self.ventana is not None:
            c_all = c_all[:, -(self.ventana + T - 1):] if c_all.shape[1] > self.ventana + T - 1 else c_all
        S = c_all.shape[1]
        q = self.q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.uk(c_all).view(B, S, H, dh).transpose(1, 2)
        v = self.uv(c_all).view(B, S, H, dh).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(dh)            # (B,H,T,S)
        if causal:
            i = torch.arange(T, device=x.device)[:, None] + (S - T)
            j = torch.arange(S, device=x.device)[None, :]
            m = j > i
            if self.ventana is not None:
                m = m | (j < i - self.ventana + 1)
            logits = logits.masked_fill(m, float("-inf"))
        if self.training:
            with torch.no_grad():
                self.max_logit = torch.maximum(self.max_logit.to(x.device),
                                               logits.masked_fill(torch.isinf(logits), -1e9).amax(dim=(0, 2, 3)).detach())
        a = torch.softmax(logits, -1)
        o = (a @ v).transpose(1, 2)                                 # (B,T,H,dh)
        o = o * torch.sigmoid(self.g(x)).view(B, T, H, dh)
        y = self.o(o.reshape(B, T, H * dh))
        keep = c_all if self.ventana is None else c_all[:, -self.ventana:]
        return y, {"c": keep}

    @torch.no_grad()
    def recorte_qk(self, tau=30.0):
        """QK-Clip por cabeza (de Kimi K2): si el logit máximo supera tau, se reescalan W_q y W_uk
        de esa cabeza por sqrt(tau / max)."""
        for h in range(self.h):
            m = float(self.max_logit[h])
            if m > tau:
                f = math.sqrt(tau / m)
                sl = slice(h * self.dh, (h + 1) * self.dh)
                self.q.weight[sl] *= f
                self.uk.weight[sl] *= f
        self.max_logit.zero_()


class StableLatentMoE(nn.Module):
    """Expertos en un espacio latente comprimido, con experto compartido y balanceo por cuantiles."""

    def __init__(self, d, n_exp=8, k=2, d_lat=None, d_exp=None, d_comp=None, momento=0.9):
        super().__init__()
        d_lat = d_lat or d // 2
        self.n, self.k = n_exp, k
        self.compartido = SiTUGLU(d, d_comp or 2 * d)
        self.baja = nn.Linear(d, d_lat, bias=False)
        self.sube = nn.Linear(d_lat, d, bias=False)
        self.norm = RMSNorm(d_lat)
        de = d_exp or 2 * d_lat
        # expertos agrupados: una contracción por proyección para los n expertos (SiTU-GLU)
        self.Wg = nn.Parameter(torch.randn(n_exp, d_lat, de) / math.sqrt(d_lat))
        self.Wu = nn.Parameter(torch.randn(n_exp, d_lat, de) / math.sqrt(d_lat))
        self.Wo = nn.Parameter(torch.randn(n_exp, de, d_lat) / math.sqrt(de))
        self.b1, self.b2 = 4.0, 25.0
        self.router = nn.Linear(d, n_exp, bias=False)
        self.register_buffer("sesgo", torch.zeros(n_exp))
        self.momento = momento
        self.ultima_carga = torch.zeros(n_exp)

    def forward(self, x):
        B, T, D = x.shape
        xs = x.reshape(-1, D)
        s = torch.sigmoid(self.router(xs))                          # (N, n)
        sel = torch.topk(s + self.sesgo, self.k, dim=-1).indices    # el sesgo solo elige
        p = torch.gather(s, 1, sel)
        p = p / (p.sum(-1, keepdim=True) + 1e-9)
        lat = self.baja(xs)
        W = torch.zeros_like(s).scatter(1, sel, p)                      # pesos solo de los elegidos
        if self.training or xs.shape[0] > 64:
            g = torch.einsum("nd,edh->neh", lat, self.Wg)
            a = self.b1 * torch.tanh(g / self.b1) * torch.sigmoid(g)
            uu = self.b2 * torch.tanh(torch.einsum("nd,edh->neh", lat, self.Wu) / self.b2)
            u = torch.einsum("neh,ehd,ne->nd", a * uu, self.Wo, W)
        else:                                                           # en vivo: solo los k elegidos
            u = torch.zeros_like(lat)
            for i in range(xs.shape[0]):
                for e in sel[i].tolist():
                    g = lat[i] @ self.Wg[e]
                    a = self.b1 * torch.tanh(g / self.b1) * torch.sigmoid(g)
                    uu = self.b2 * torch.tanh(lat[i] @ self.Wu[e] / self.b2)
                    u[i] = u[i] + W[i, e] * ((a * uu) @ self.Wo[e])
        y = self.compartido(xs) + self.sube(self.norm(u))
        if self.training:
            with torch.no_grad():
                self._balancear(s.detach())
                self.ultima_carga = torch.bincount(sel.flatten(), minlength=self.n).float() / sel.numel()
        return y.view(B, T, D)

    @torch.no_grad()
    def _balancear(self, s):
        """Quantile Balancing: b̂_j = -cuantil_{1-k/n}(s_{:,j} - α), α = umbral de elección por
        ficha; luego se centra. Con media móvil para estabilidad."""
        umbral = torch.topk(s + self.sesgo, self.k, dim=-1).values[:, -1:]      # α por ficha
        q = 1.0 - self.k / self.n
        if s.shape[0] > 50000:                                                 # cuantil sobre una muestra
            s, umbral = s[::s.shape[0] // 50000 + 1], umbral[::s.shape[0] // 50000 + 1]
        bh = -torch.quantile(s - umbral, q, dim=0)    # se elige j si s_j - α >= -b_j: carga = k/n
        bh = bh - bh.mean()
        self.sesgo.mul_(self.momento).add_((1 - self.momento) * bh)
        self.sesgo.sub_(self.sesgo.mean())


class AttnResBloque(nn.Module):
    """Atención sobre la profundidad (Block AttnRes): una pseudo-consulta por subcapa."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.norm = RMSNorm(d, afin=False)

    def forward(self, valores):
        V = torch.stack(valores, 0)                                 # (n, B, T, D)
        logits = torch.einsum("d,nbtd->nbt", self.w, self.norm(V))
        return torch.einsum("nbt,nbtd->btd", torch.softmax(logits, 0), V)


class Bloque(nn.Module):
    """Mezclador (KDA o MLA) + Stable LatentMoE, cada uno con su AttnRes de entrada."""

    def __init__(self, d, tipo, cabezas=4, dk=16, n_exp=8, k=2, ventana=None):
        super().__init__()
        self.tipo = tipo
        self.res1, self.res2 = AttnResBloque(d), AttnResBloque(d)
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mix = KDA(d, cabezas, dk, dk) if tipo == "kda" else GatedMLA(d, cabezas, dk, d_lat=2 * dk, ventana=ventana)
        self.moe = StableLatentMoE(d, n_exp, k)

    def forward(self, bloques, parcial, estado=None, causal=True):
        h = self.res1(bloques + [parcial])
        if self.tipo == "kda":
            y, est = self.mix(self.n1(h), estado)
        else:
            y, est = self.mix(self.n1(h), estado, causal=causal)
        parcial = parcial + y
        h = self.res2(bloques + [parcial])
        parcial = parcial + self.moe(self.n2(h))
        return parcial, est
