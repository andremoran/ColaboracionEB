"""Optimizador Muon (ortogonalización de Newton–Schulz del momento) con partición por cabeza,
como en Kimi K3 ("Per-Head Muon"), más AdamW para embebidos, normas, sesgos y cabezas de salida.

Escala de la actualización: 0.2·sqrt(max(m, n)), que iguala el RMS de la actualización al de AdamW
(receta de Moonlight/Kimi). El recorte QK de las capas MLA se aplica aparte tras cada paso.
"""
import math

import torch


@torch.no_grad()
def newton_schulz5(G, pasos=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    trans = X.shape[-2] > X.shape[-1]
    if trans:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(pasos):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if trans:
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momento=0.95, nesterov=True, decaimiento=0.01):
        super().__init__(params, dict(lr=lr, momento=momento, nesterov=nesterov, decaimiento=decaimiento, cabezas=1))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "M" not in st:
                    st["M"] = torch.zeros_like(p)
                M = st["M"]
                M.mul_(g["momento"]).add_(p.grad)
                G = p.grad.add(M, alpha=g["momento"]) if g["nesterov"] else M
                h = getattr(p, "_cabezas", 1)
                if h > 1 and G.shape[0] % h == 0:
                    U = newton_schulz5(G.view(h, G.shape[0] // h, G.shape[1])).reshape_as(G)
                    esc = 0.2 * math.sqrt(max(G.shape[0] // h, G.shape[1]))
                else:
                    U = newton_schulz5(G)
                    esc = 0.2 * math.sqrt(max(G.shape))
                p.mul_(1 - g["lr"] * g["decaimiento"])
                p.add_(U, alpha=-g["lr"] * esc)


def repartir_parametros(modelo):
    """Matrices ocultas -> Muon (por cabeza en q/k/v/o/g de las atenciones); el resto -> AdamW."""
    muon, adam = [], []
    for nombre, p in modelo.named_parameters():
        if not p.requires_grad:
            continue
        es_oculta = p.ndim == 2 and not any(s in nombre for s in ("tok.emb", "tok.obj", "tok.modo", "tok.sitio",
                                                                    "h_", "d_in", "router", ".z."))
        if es_oculta and ("enc." in nombre or "dec." in nombre):
            partes = nombre.split(".")
            if any(x in partes for x in ("q", "k", "v", "g", "uk", "uv")) and partes[-1] == "weight":
                mod = modelo.get_submodule(".".join(partes[:-2]))
                p._cabezas = getattr(mod, "h", 1)
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam
