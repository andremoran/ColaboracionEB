"""Puente entre el esqueleto del Avatar 3D y el modelo musculoesquelético MyoLeg (MuJoCo, 80 músculos).

- Coordenadas: de las rotaciones locales anatómicas de cadera, rodilla y tobillo (las mismas que usa
  el generador y el reajuste) a hip_flexion / hip_adduction / hip_rotation / knee_angle / ankle_angle /
  subtalar_angle de MyoLeg, con las coordenadas acopladas de la rodilla por sus polinomios.
- Por fotograma: brazos de momento r_ij (N·m/N), longitudes y velocidades de fibra-tendón, fuerza
  máxima isométrica F0 y multiplicadores activos fuerza-longitud y fuerza-velocidad de MuJoCo.
- Signos: fuerza generalizada positiva = tiende a aumentar la coordenada. Con la convención de marcha
  de nucleo.dinamica (extensor +): τ_hip_flexion = -M_ext_cadera, τ_knee = -M_ext_rodilla,
  τ_ankle = -M_plantar, τ_hip_adduction = -M_abd, τ_hip_rotation = +M_rotint.
"""
from functools import lru_cache

import mujoco
import numpy as np

from nucleo.rotaciones import seis_d_a_matriz

DOFS = ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r", "subtalar_angle_r",
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l", "subtalar_angle_l"]
# momentos anatómicos (nucleo.dinamica) -> fuerza generalizada de cada DOF (signo)
MOMENTO_A_DOF = {"hip_flexion_r": ("caderaD_ext", -1), "hip_adduction_r": ("caderaD_abd", -1),
                 "hip_rotation_r": ("caderaD_rotint", 1), "knee_angle_r": ("rodillaD_ext", -1),
                 "ankle_angle_r": ("tobilloD_plantar", -1), "subtalar_angle_r": (None, 0),
                 "hip_flexion_l": ("caderaI_ext", -1), "hip_adduction_l": ("caderaI_abd", -1),
                 "hip_rotation_l": ("caderaI_rotint", 1), "knee_angle_l": ("rodillaI_ext", -1),
                 "ankle_angle_l": ("tobilloI_plantar", -1), "subtalar_angle_l": (None, 0)}


@lru_cache(maxsize=1)
def modelo():
    import myo_sim
    m, d = myo_sim.load("myolegs")
    return m, d


def _euler_cadera(R, s):
    """Inversa de R = Rx(-flex)·Rz(s·abd)·Ry(-s·rot) (grados)."""
    # R = Rx(a) Rz(b) Ry(c): R[0,1] = -sin b ; R[2,1] = cos b·sin a ; R[1,1] = cos a cos b ;
    # R[0,0] = cos b cos c ; R[0,2] = cos b sin c
    b = np.arcsin(np.clip(-R[..., 0, 1], -1, 1))
    a = np.arctan2(R[..., 2, 1], R[..., 1, 1])
    c = np.arctan2(R[..., 0, 2], R[..., 0, 0])
    flex, abd, rot = -np.degrees(a), s * np.degrees(b), -s * np.degrees(c)
    return flex, abd, rot


def _euler_tobillo(R, s):
    """Inversa de R = Rx(-dorsi)·Rz(-s·inv)."""
    b = np.arcsin(np.clip(-R[..., 0, 1], -1, 1))
    a = np.arctan2(R[..., 2, 1], R[..., 1, 1])
    return -np.degrees(a), -s * np.degrees(b)


def coordenadas(R6):
    """R6 (T,22,6) rotaciones locales del avatar -> dict DOF de MyoLeg -> (T,) en radianes."""
    R = seis_d_a_matriz(np.asarray(R6, dtype=np.float64))
    q = {}
    for lado, s, jc, jr, jt in (("r", -1.0, 2, 5, 8), ("l", 1.0, 1, 4, 7)):
        flex, abd, rot = _euler_cadera(R[:, jc], s)
        kne = np.degrees(np.arctan2(R[:, jr, 2, 1], R[:, jr, 1, 1]))    # Rx(+k)
        dor, inv = _euler_tobillo(R[:, jt], s)
        q[f"hip_flexion_{lado}"] = np.radians(flex)
        q[f"hip_adduction_{lado}"] = np.radians(-abd)
        q[f"hip_rotation_{lado}"] = np.radians(rot)
        q[f"knee_angle_{lado}"] = np.radians(np.clip(kne, 0, 125))
        q[f"ankle_angle_{lado}"] = np.radians(dor)
        q[f"subtalar_angle_{lado}"] = np.radians(inv)
    return q


CLAVES_MOMENTOS = ['caderaD_abd', 'caderaD_ext', 'caderaD_rotint', 'caderaI_abd', 'caderaI_ext', 'caderaI_rotint',
                   'codoD_flex', 'codoI_flex', 'cuello_ext', 'hombroD_abd', 'hombroD_flex', 'hombroD_rotint',
                   'hombroI_abd', 'hombroI_flex', 'hombroI_rotint', 'lumbar_ext', 'lumbar_lat', 'lumbar_rot',
                   'munecaD_flex', 'munecaI_flex', 'rodillaD_ext', 'rodillaI_ext', 'tobilloD_plantar', 'tobilloI_plantar']


def fuerzas_generalizadas(R6, mom):
    """Momentos articulares (T,24; orden CLAVES_MOMENTOS; componentes en el marco de la pelvis para la
    cadera) -> fuerzas generalizadas (T,12) de las coordenadas de MyoLeg.

    La cadera de MyoLeg (como la de OpenSim) encadena flexión -> aducción -> rotación: el eje de la
    aducción gira con la flexión y el de la rotación es el eje largo del fémur. La fuerza generalizada
    de cada coordenada es la proyección del momento sobre SU eje, τ_i = e_i · M (τ = Jᵀ M). Con la cadera
    flexionada 90° la aducción recibe la componente vertical de la pelvis y la rotación la
    anteroposterior; proyectar en los ejes fijos de la pelvis las intercambiaba."""
    R = seis_d_a_matriz(np.asarray(R6, dtype=np.float64))
    mom = np.asarray(mom, dtype=np.float64)
    c = {k: mom[:, i] for i, k in enumerate(CLAVES_MOMENTOS)}
    tau = np.zeros((len(mom), len(DOFS)))
    for lado, L, s, jc in (("r", "D", -1.0, 2), ("l", "I", 1.0, 1)):
        flex, abd, _ = _euler_cadera(R[:, jc], s)
        a, b = -np.radians(flex), s * np.radians(abd)
        # componentes del momento de cadera en el marco de la pelvis (inversa de momentos_anatomicos)
        mx, my, mz = c[f"cadera{L}_ext"], -s * c[f"cadera{L}_rotint"], s * c[f"cadera{L}_abd"]
        tau[:, DOFS.index(f"hip_flexion_{lado}")] = -mx
        tau[:, DOFS.index(f"hip_adduction_{lado}")] = -s * (-np.sin(a) * my + np.cos(a) * mz)
        tau[:, DOFS.index(f"hip_rotation_{lado}")] = -s * (-np.sin(b) * mx + np.cos(a) * np.cos(b) * my
                                                          + np.sin(a) * np.cos(b) * mz)
        tau[:, DOFS.index(f"knee_angle_{lado}")] = -c[f"rodilla{L}_ext"]
        tau[:, DOFS.index(f"ankle_angle_{lado}")] = -c[f"tobillo{L}_plantar"]
    return tau


def qpos(m, qd):
    """Vector qpos de MyoLeg (una fila por fotograma) con la rodilla acoplada por polinomios."""
    T = len(next(iter(qd.values())))
    Q = np.tile(m.qpos0, (T, 1))
    for n, v in qd.items():
        j = m.joint(n).id
        rng = m.jnt_range[j]
        Q[:, m.jnt_qposadr[j]] = np.clip(v, rng[0], rng[1])
    for e in range(m.neq):
        if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
        x = Q[:, m.jnt_qposadr[j2]]
        c = m.eq_data[e, :5]
        Q[:, m.jnt_qposadr[j1]] = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4
    return Q


def qvel(m, Q, fps):
    """Velocidades generalizadas por diferencia central en la variedad (la raíz tiene cuaternión)."""
    T = len(Q)
    V = np.zeros((T, m.nv))
    for t in range(T):
        a, b = max(0, t - 1), min(T - 1, t + 1)
        if b > a:
            mujoco.mj_differentiatePos(m, V[t], (b - a) / fps, Q[a], Q[b])
    return V


class Musculos:
    """Brazos de momento, longitudes y capacidades de fuerza de los 80 músculos de MyoLeg."""

    def __init__(self):
        self.m, self.d = modelo()
        m = self.m
        self.nombres = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.dof_idx = [m.jnt_dofadr[m.joint(n).id] for n in DOFS]
        self.F0 = m.actuator_gainprm[:, 2].copy()
        self.prm = m.actuator_gainprm[:, :9].copy()
        self.lr = m.actuator_lengthrange.copy()

    def geometria(self, Q, Qd=None):
        """Q (T,nq) -> brazos R (T,nu,12 DOF), longitud (T,nu), velocidad (T,nu)."""
        m, d = self.m, self.d
        T = len(Q)
        R = np.zeros((T, m.nu, len(DOFS)))
        Lm = np.zeros((T, m.nu))
        Vm = np.zeros((T, m.nu))
        for t in range(T):
            d.qpos[:] = Q[t]
            d.qvel[:] = 0 if Qd is None else Qd[t]
            mujoco.mj_fwdPosition(m, d)
            mujoco.mj_fwdVelocity(m, d)
            mom = np.zeros((m.nu, m.nv))
            mujoco.mju_sparse2dense(mom, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
            R[t] = mom[:, self.dof_idx]
            Lm[t] = d.actuator_length
            Vm[t] = d.actuator_velocity
        return R, Lm, Vm

    def capacidad(self, Lm, Vm):
        """Fuerza activa máxima F0·FL·FV y fuerza pasiva por músculo (curvas de MuJoCo)."""
        T, nu = Lm.shape
        Fa = np.zeros((T, nu))
        Fp = np.zeros((T, nu))
        for i in range(nu):
            p = self.prm[i]
            L0 = (self.lr[i, 1] - self.lr[i, 0]) / (p[1] - p[0])
            ln = p[0] + (Lm[:, i] - self.lr[i, 0]) / L0          # longitud normalizada
            vn = Vm[:, i] / (L0 * p[6])                         # velocidad normalizada (vmax)
            fl = _curva_fl(ln, p[4], p[5])
            fv = _curva_fv(vn, p[8])
            fp = _curva_pasiva(ln, p[5], p[7])
            Fa[:, i] = self.F0[i] * fl * fv
            Fp[:, i] = self.F0[i] * fp
        return Fa, Fp


def _curva_fl(L, lmin, lmax):
    """Fuerza-longitud activa de MuJoCo (dos parábolas por tramos, pico en L=1)."""
    a = 0.5 * (lmin + 1)
    b = 0.5 * (1 + lmax)
    x = np.zeros_like(L)
    m1 = (L >= lmin) & (L <= a)
    x[m1] = 0.5 * ((L[m1] - lmin) / (a - lmin)) ** 2
    m2 = (L > a) & (L <= 1)
    x[m2] = 1 - 0.5 * ((1 - L[m2]) / (1 - a)) ** 2
    m3 = (L > 1) & (L <= b)
    x[m3] = 1 - 0.5 * ((L[m3] - 1) / (b - 1)) ** 2
    m4 = (L > b) & (L <= lmax)
    x[m4] = 0.5 * ((lmax - L[m4]) / (lmax - b)) ** 2
    return np.clip(x, 0.02, 1)


def _curva_fv(V, fvmax):
    y = fvmax - 1
    out = np.where(V <= -1, 0.0, np.where(V <= 0, (V + 1) ** 2, np.where(V <= y, fvmax - (y - V) ** 2 / y, fvmax)))
    return np.clip(out, 0.05, fvmax)


def _curva_pasiva(L, lmax, fpmax):
    b = 0.5 * (1 + lmax)
    return np.where(L <= 1, 0.0, np.where(L <= b, fpmax * 0.25 * ((L - 1) / (b - 1)) ** 3,
                                          fpmax * (0.25 + 0.75 * (L - b) / (b - 1))))
