"""K3Mov en vivo: un fotograma a la vez, con estado.

El codificador y el decodificador son causales. En vivo cada capa KDA conserva su memoria S (y el
estado de sus convoluciones cortas) y cada capa MLA su caché latente de claves y valores, limitada a
una ventana igual a la de entrenamiento; así el coste por fotograma es constante. La fase se obtiene
con el filtro hacia delante (Minería VII, P1) sobre las probabilidades del modelo, restringido al orden
de fases del ejercicio más probable, y la repetición se cuenta sumando la densidad.
"""
import time

import numpy as np
import torch

from .decodificacion import matriz_transicion
from .modelo import NOMBRES_CELDAS
from nucleo.ejercicios import EJERCICIOS, NOMBRES_EJ


class K3MovEnVivo:
    def __init__(self, modelo, ventana=128):
        self.m = modelo.eval()
        for mla in self.m.capas_mla():
            mla.ventana = ventana
        self.reiniciar()

    def reiniciar(self):
        self.est_enc = None
        self.est_dec = None
        self.alfa = None                      # creencia del filtro de fases
        self.conteo = 0.0
        self.p_ej_acum = np.zeros(17)
        self.lat_ms = []

    @torch.no_grad()
    def paso(self, celdas, presentes, decodificar=True):
        """celdas: dict nombre -> (dim,) de UN fotograma; presentes: (C,) bool."""
        return self.bloque({k: np.asarray(v)[None] for k, v in celdas.items()}, np.asarray(presentes)[None], decodificar)

    @torch.no_grad()
    def bloque(self, celdas, presentes, decodificar=True):
        """Varios fotogramas seguidos de una vez (celdas: nombre -> (n, dim); presentes: (n, C)): el
        resultado es idéntico al de procesarlos uno a uno y el coste fijo por llamada se reparte.
        Devuelve el estado tras el último fotograma: ejercicio probable, fase filtrada, conteo acumulado,
        probabilidades de error y, si se pide, la postura correcta (22 rotaciones 6D por fotograma)."""
        t0 = time.perf_counter()
        n = len(presentes)
        cel = {k: torch.as_tensor(np.asarray(celdas[k], np.float32)).view(1, n, -1) for k in NOMBRES_CELDAS}
        P = torch.as_tensor(np.asarray(presentes, bool)).view(1, n, -1)
        h, _, self.est_enc = self.m.codificar(cel, P, self.est_enc, causal=True)
        out = self.m.cabezas(h)
        rot = None
        if decodificar:
            dec, self.est_dec = self.m.decodificar(h, out["ejercicio"], cel, self.est_dec, causal=True)
            rot = dec["rot6"][0].numpy()
        p_ej = torch.softmax(out["ejercicio"][0], -1).numpy()
        p_f = torch.softmax(out["fase"][0], -1).numpy()
        for t in range(n):
            self.p_ej_acum = 0.97 * self.p_ej_acum + 0.03 * p_ej[t]     # memoria del ejercicio (~1 s)
            ej = int(self.p_ej_acum[:16].argmax())
            Tm = matriz_transicion(EJERCICIOS[NOMBRES_EJ[ej]]["primera"])
            pred = p_f[t] if self.alfa is None else (self.alfa @ Tm) * p_f[t]   # filtro hacia delante
            self.alfa = pred / max(pred.sum(), 1e-12)
        self.conteo += float(out["rep"][0].sum())
        self.lat_ms.append((time.perf_counter() - t0) * 1000 / n)
        conf = float(self.p_ej_acum[ej] / max(self.p_ej_acum.sum(), 1e-9))      # probabilidad normalizada
        return dict(ejercicio=NOMBRES_EJ[ej], p_ejercicio=conf, fase=int(self.alfa.argmax()),
                    repeticiones=self.conteo, errores=torch.sigmoid(out["errores"][0, -1]).numpy(), rot6_correcto=rot)
