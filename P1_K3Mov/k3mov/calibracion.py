"""Calibración de las probabilidades de error de técnica (escalado de Platt por tipo de error).

K3Mov ordena bien las repeticiones con y sin error (AUC), pero sus probabilidades salen bajas: casi
nunca superan 0,5. Con la partición de VALIDACIÓN se ajusta, para cada tipo de error,
p = σ(a·logit(s) + b) sobre el puntaje s del modelo; los tipos con pocos ejemplos usan el ajuste común.
Dos niveles: repetición (media de la probabilidad dentro de la repetición) y serie (percentil 90 en
toda la serie, que es lo que muestra la aplicación). Una transformación creciente por tipo no cambia el
AUC de ese tipo; sí hace que 0,5 signifique «más probable que no».
"""
import json
from pathlib import Path

import numpy as np

ARCHIVO = "calibracion_errores.json"


def _logit(s):
    s = np.clip(np.asarray(s, np.float64), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def ajustar(y, s, tipos, min_ejemplos=8):
    """y (n,) 0/1, s (n,) puntajes del modelo, tipos (n,) nombre del error -> parámetros de Platt."""
    from sklearn.linear_model import LogisticRegression
    y, x, tipos = np.asarray(y, int), _logit(s)[:, None], np.asarray(tipos)

    def platt(yy, xx):
        m = LogisticRegression(C=10.0).fit(xx, yy)
        return [float(m.coef_[0, 0]), float(m.intercept_[0])]
    out = dict(comun=platt(y, x), por_tipo={})
    for t in sorted(set(tipos.tolist())):
        k = tipos == t
        if (y[k] == 1).sum() >= min_ejemplos and (y[k] == 0).sum() >= min_ejemplos:
            out["por_tipo"][t] = platt(y[k], x[k])
    return out


def aplicar(cal, tipo, s):
    """Probabilidad calibrada del error `tipo` (clave del catálogo, p. ej. 'sentadilla:rodillas_valgo')."""
    a, b = cal["por_tipo"].get(tipo, cal["comun"])
    return 1.0 / (1.0 + np.exp(-(a * _logit(s) + b)))


def cargar(ruta_modelo):
    """Calibración guardada junto al modelo (o None si no se ajustó)."""
    p = Path(ruta_modelo).with_name(ARCHIVO)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
