"""Post-proceso de K3Mov: fases por la explicación más probable (Viterbi) o por filtrado en vivo,
repeticiones por suma de la densidad y sus picos, y tempos concéntrico/excéntrico por repetición.

La sucesión de fases de un ejercicio es un proceso con orden: reposo -> (1ª fase de movimiento) ->
pausa -> (2ª fase) -> pausa -> ... -> reposo. El orden depende del ejercicio (sentadilla empieza
excéntrica; curl, concéntrica). La explicación más probable no es la sucesión de los estados más
probables de cada instante (esa puede tener probabilidad nula): por eso se decodifica con Viterbi.
"""
import numpy as np

from nucleo.ejercicios import EJERCICIOS, NOMBRES_EJ

# 0 reposo · 1 concéntrica · 2 pausa contraída · 3 excéntrica · 4 pausa estirada
SIG = {"exc": {3: 4, 4: 1, 1: 2, 2: 3}, "con": {1: 2, 2: 3, 3: 4, 4: 1}}


def matriz_transicion(primera, quedarse=0.92):
    """Transiciones permitidas; las pausas pueden saltarse (duración cero)."""
    Tm = np.full((5, 5), 1e-6)
    if primera == "iso":
        Tm[0, 0], Tm[0, 2], Tm[2, 2], Tm[2, 0] = 0.97, 0.03, 0.99, 0.01
        return Tm / Tm.sum(1, keepdims=True)
    sig = SIG[primera]
    inicio = 3 if primera == "exc" else 1
    Tm[0, 0] = 0.97
    Tm[0, inicio] = 0.03
    for s, n in sig.items():
        Tm[s, s] = quedarse
        Tm[s, n] = (1 - quedarse) * 0.8
        if s in (4, 2):                      # desde una pausa se puede volver al reposo (fin de serie)
            Tm[s, 0] = (1 - quedarse) * 0.2
        else:                                # saltar la pausa siguiente
            Tm[s, sig[n]] = (1 - quedarse) * 0.2
    return Tm / Tm.sum(1, keepdims=True)


def viterbi(logp, Tm, prior=None):
    """logp (T,5) log-probabilidades por fotograma -> sucesión de estados más probable."""
    T, S = logp.shape
    lT = np.log(Tm)
    d = np.log(prior if prior is not None else np.array([0.9, 0.025, 0.025, 0.025, 0.025])) + logp[0]
    bp = np.zeros((T, S), int)
    for t in range(1, T):
        c = d[:, None] + lT
        bp[t] = c.argmax(0)
        d = c.max(0) + logp[t]
    x = np.zeros(T, int)
    x[-1] = d.argmax()
    for t in range(T - 1, 0, -1):
        x[t - 1] = bp[t, x[t]]
    return x


def filtro_en_vivo(logp, Tm):
    """Filtrado hacia delante (mensaje f_{1:t}): coste O(S²) por fotograma, sin mirar el futuro."""
    T, S = logp.shape
    f = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
    out = np.zeros(T, int)
    for t in range(T):
        f = (f @ Tm) * np.exp(logp[t] - logp[t].max())
        f /= f.sum()
        out[t] = f.argmax()
    return out


def picos_repeticion(dens, min_sep=8, umbral=0.25):
    """Instantes de fin de repetición = máximos locales de la densidad suavizada."""
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    k /= k.sum()
    s = np.convolve(dens, k, mode="same")
    cand = [t for t in range(1, len(s) - 1) if s[t] >= s[t - 1] and s[t] > s[t + 1] and s[t] > umbral * 0.2]
    # masa en torno a cada pico: un pico por repetición completa
    elegidos = []
    for t in sorted(cand, key=lambda t: -s[t]):
        if all(abs(t - u) >= min_sep for u in elegidos):
            masa = dens[max(0, t - min_sep // 2): t + min_sep // 2 + 1].sum()
            if masa >= umbral:
                elegidos.append(t)
    return sorted(elegidos)


def tempos(fases, fps, fines):
    """Duración concéntrica y excéntrica de cada repetición (entre fines consecutivos)."""
    res = []
    bordes = [0] + [f + 1 for f in fines]
    for i in range(1, len(bordes)):
        seg = fases[bordes[i - 1]: bordes[i]]
        res.append(dict(fin=bordes[i] - 1, t_con=float((seg == 1).sum() / fps), t_exc=float((seg == 3).sum() / fps),
                        t_pausas=float(((seg == 2) | (seg == 4)).sum() / fps)))
    return res


def fines_por_fases(fases, primera, min_largo=8):
    """Fin de cada repetición según la sucesión de fases: el final de la SEGUNDA fase de movimiento
    (concéntrica si el ejercicio empieza bajando, excéntrica si empieza subiendo). Respaldo cuando la
    densidad de repetición no marca picos claros."""
    segunda = 1 if primera == "exc" else 3
    fines, t = [], 0
    T = len(fases)
    while t < T:
        if fases[t] == segunda:
            a = t
            while t + 1 < T and fases[t + 1] == segunda:
                t += 1
            if t - a + 1 >= min_largo // 2:
                fines.append(t)
        t += 1
    return fines


def decodificar_serie(probs_fase, prob_ej, dens, fps=30.0, en_vivo=False):
    """Resumen de una serie: ejercicio, fases, repeticiones y tempos."""
    act = prob_ej[:, :-1].sum(0)
    ej = int(act.argmax())
    primera = EJERCICIOS[NOMBRES_EJ[ej]]["primera"]
    Tm = matriz_transicion(primera)
    lp = np.log(np.clip(probs_fase, 1e-6, 1))
    fases = filtro_en_vivo(lp, Tm) if en_vivo else viterbi(lp, Tm)
    fines = picos_repeticion(dens)
    if primera != "iso" and len(fines) < round(float(dens.sum())) - 1:
        fines = fines_por_fases(fases, primera)
    return dict(ejercicio=NOMBRES_EJ[ej], conteo=float(dens.sum()), n_picos=len(fines), fines=fines, fases=fases,
                tempos=tempos(fases, fps, fines))
