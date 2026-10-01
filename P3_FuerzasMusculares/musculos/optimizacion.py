"""Optimización estática y su variante guiada por EMG.

Por fotograma, con tensión F_i = a_i·Fa_i + Fp_i (Fa = F0·FL·FV) y brazos de momento r_ij:
    min  Σ_i a_i² + w_r Σ_j ρ_j²  (+ w_e Σ_{i∈EMG} (a_i - g_i·e_i)²)
    s.a. -Σ_i r_ij F_i + ρ_j = τ_j   (ρ = actuador de reserva, muy penalizado)
         0 ≤ a_i ≤ 1
Se resuelve como mínimos cuadrados acotados (la igualdad entra con peso alto), en paralelo por
fotograma. La suma de activaciones al cuadrado reparte la carga entre sinergistas en proporción a
su capacidad, criterio clásico de mínimo esfuerzo.
"""
import numpy as np
from scipy.optimize import lsq_linear


def resolver(R, Fa, Fp, tau, w_igualdad=50.0, w_reserva=0.02, emg=None, w_emg=4.0, esfuerzo=None):
    """R (T,nu,nd), Fa/Fp (T,nu), tau (T,nd) -> a (T,nu), F (T,nu), reserva (T,nd).
    emg: dict índice de músculo -> (T,) excitación normalizada 0-1 (opcional)."""
    T, nu, nd = R.shape
    a = np.zeros((T, nu))
    res = np.zeros((T, nd))
    peso = np.ones(nu) if esfuerzo is None else esfuerzo
    lb = np.r_[np.zeros(nu), -np.inf * np.ones(nd)]
    ub = np.r_[np.ones(nu), np.inf * np.ones(nd)]
    x0 = None
    for t in range(T):
        A_eq = np.c_[-(R[t] * Fa[t][:, None]).T, np.eye(nd)]          # (nd, nu+nd)
        b_eq = tau[t] + (R[t] * Fp[t][:, None]).sum(0)
        filas = [w_igualdad * A_eq, np.c_[np.diag(np.sqrt(peso)), np.zeros((nu, nd))],
                 np.c_[np.zeros((nd, nu)), np.sqrt(w_reserva) * np.eye(nd)]]
        lados = [w_igualdad * b_eq, np.zeros(nu), np.zeros(nd)]
        if emg:
            for i, e in emg.items():
                fila = np.zeros(nu + nd)
                fila[i] = np.sqrt(w_emg)
                filas.append(fila[None])
                lados.append(np.array([np.sqrt(w_emg) * e[t]]))
        A = np.vstack(filas)
        b = np.concatenate(lados)
        sol = lsq_linear(A, b, bounds=(lb, ub), method="bvls" if nu + nd < 150 else "trf", lsmr_tol="auto")
        a[t] = sol.x[:nu]
        res[t] = sol.x[nu:]
    F = a * Fa + Fp
    return a, F, res


def reproducido(R, F):
    """Momento generalizado que producen las tensiones: -Σ r_ij F_i."""
    return -np.einsum("tij,ti->tj", R, F)


# ---------------------------------------------------------------- fatiga ---
def fatiga_tres_compartimentos(a, fps=30.0, F=0.0146, Rr=0.00149, L=10.0):
    """Modelo de tres compartimentos (activo MA, fatigado MF, en reposo MR) por músculo.
    La orden central C sigue a la demanda (activación) con ganancia L; F y R son las tasas de fatiga
    y de recuperación (s⁻¹, valores de referencia para músculos de extremidad). Devuelve la fracción de
    capacidad residual (1 - MF) por fotograma."""
    T, n = a.shape
    dt = 1.0 / fps
    MA = np.zeros(n); MF = np.zeros(n); MR = np.ones(n)
    res = np.zeros((T, n))
    for t in range(T):
        obj = a[t]
        C = np.where(MA < obj, np.where(MR > obj - MA, L * (obj - MA), L * MR), L * (obj - MA))
        dMA = C - F * MA
        dMF = F * MA - Rr * MF
        dMR = -C + Rr * MF
        MA = np.clip(MA + dt * dMA, 0, 1); MF = np.clip(MF + dt * dMF, 0, 1); MR = np.clip(MR + dt * dMR, 0, 1)
        s = MA + MF + MR
        MA, MF, MR = MA / s, MF / s, MR / s
        res[t] = 1 - MF
    return res
