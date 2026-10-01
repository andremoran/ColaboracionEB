"""Arquitecturas del Proyecto 2.

B · TCN «caja negra»: convoluciones temporales dilatadas (residuales) de la captura a los momentos,
    reacciones y potencias. La convolución es la capa lineal forzada por la invariancia a la
    traslación en el tiempo (el mismo gesto vale lo mismo empiece cuando empiece).
C · Híbrido: la misma TCN estima el ESTADO cinemático limpio (orientaciones 6D de los 22 segmentos y
    trayectoria de la pelvis); los momentos los calcula después el Newton–Euler validado contra
    MuJoCo. La red solo aprende lo que la física no da (quitar el ruido de la captura).
D · Red lagrangiana profunda (DeLaN) en el plano sagital con el pie como base fija:
    τ = M(q) q̈ + C(q, q̇) q̇ + ∂V/∂q, con M = L Lᵀ (L triangular con diagonal positiva, siempre
    definida positiva) y V una red escalar; el término de Coriolis sale por diferenciación
    automática. Se compara con un perceptrón sin estructura física.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BloqueTCN(nn.Module):
    def __init__(self, c, dil, k=5, causal=False):
        super().__init__()
        self.pad = (k - 1) * dil if causal else (k - 1) * dil // 2
        self.causal = causal
        self.c1 = nn.Conv1d(c, c, k, dilation=dil)
        self.c2 = nn.Conv1d(c, c, 1)
        self.n = nn.GroupNorm(1, c)

    def forward(self, x):
        h = F.pad(x, (self.pad, 0) if self.causal else (self.pad, self.pad))
        h = self.c2(F.gelu(self.c1(h)))
        return x + self.n(h)


class TCN(nn.Module):
    def __init__(self, f_in, salidas, c=96, niveles=5, causal=False):
        super().__init__()
        self.inp = nn.Conv1d(f_in, c, 1)
        self.bloques = nn.Sequential(*[BloqueTCN(c, 2 ** i, causal=causal) for i in range(niveles)])
        self.cabezas = nn.ModuleDict({k: nn.Conv1d(c, d, 1) for k, d in salidas.items()})

    def forward(self, X):
        h = self.bloques(self.inp(X.transpose(1, 2)))
        return {k: m(h).transpose(1, 2) for k, m in self.cabezas.items()}


# =============================================================== DeLaN ===
class DeLaN(nn.Module):
    """Red lagrangiana profunda (Lutter, Ritter y Peters, 2019) para n grados de libertad."""

    def __init__(self, n, oculto=64, disipacion=True):
        super().__init__()
        self.n = n
        self.tronco = nn.Sequential(nn.Linear(2 * n, oculto), nn.Softplus(), nn.Linear(oculto, oculto), nn.Softplus())
        self.l_diag = nn.Linear(oculto, n)
        self.l_bajo = nn.Linear(oculto, n * (n - 1) // 2)
        self.V = nn.Sequential(nn.Linear(2 * n, oculto), nn.Softplus(), nn.Linear(oculto, 1))
        self.D = nn.Parameter(torch.zeros(n)) if disipacion else None
        self.idx = torch.tril_indices(n, n, -1)

    def _feat(self, q):
        """Orientación absoluta de cada segmento de la cadena (suma acumulada de ángulos relativos)."""
        qa = torch.cumsum(q, -1)
        return torch.cat([torch.sin(qa), torch.cos(qa)], -1)

    def masa(self, q):
        h = self.tronco(self._feat(q))
        Ld = F.softplus(self.l_diag(h)) + 1e-3
        L = torch.diag_embed(Ld)
        L = L.index_put((torch.arange(q.shape[0])[:, None].expand(-1, self.idx.shape[1]),
                         self.idx[0].expand(q.shape[0], -1), self.idx[1].expand(q.shape[0], -1)), self.l_bajo(h))
        return L @ L.transpose(-1, -2)

    def forward(self, q, qd, qdd):
        """τ = M q̈ + Ṁ q̇ - ½ ∂(q̇ᵀ M q̇)/∂q + ∂V/∂q + D q̇."""
        q = q.detach().requires_grad_(True)
        M = self.masa(q)
        # Ṁ q̇ por derivada direccional (jvp) y ∂(q̇ᵀMq̇)/∂q por gradiente
        cin = 0.5 * torch.einsum("bi,bij,bj->b", qd, M, qd)
        dcin_dq = torch.autograd.grad(cin.sum(), q, create_graph=True)[0]
        Mqd = torch.einsum("bij,bj->bi", M, qd)
        dMqd = torch.stack([torch.autograd.grad(Mqd[:, i].sum(), q, create_graph=True)[0] for i in range(self.n)], 1)
        Mdot_qd = torch.einsum("bij,bj->bi", dMqd, qd)
        Vq = self.V(self._feat(q)).sum()
        dV = torch.autograd.grad(Vq, q, create_graph=True)[0]
        tau = torch.einsum("bij,bj->bi", M, qdd) + Mdot_qd - dcin_dq + dV
        if self.D is not None:
            tau = tau + F.softplus(self.D) * qd
        return tau, M

    def energia(self, q, qd):
        M = self.masa(q)
        return 0.5 * torch.einsum("bi,bij,bj->b", qd, M, qd), self.V(self._feat(q)).squeeze(-1)


class PerceptronID(nn.Module):
    def __init__(self, n, oculto=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * n, oculto), nn.GELU(), nn.Linear(oculto, oculto), nn.GELU(), nn.Linear(oculto, n))

    def forward(self, q, qd, qdd):
        qa = torch.cumsum(q, -1)
        return self.net(torch.cat([torch.sin(qa), torch.cos(qa), qd, qdd], -1)), None
