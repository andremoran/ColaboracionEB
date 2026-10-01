"""Validación de la dinámica inversa Newton–Euler (nucleo.dinamica) contra MuJoCo (mj_inverse).

Se monta en MJCF el MISMO cuerpo (22 segmentos, masas, centros de masa e inercias de de Leva)
con articulaciones esféricas y raíz libre; se impone un movimiento suave, se calcula la dinámica
inversa con los dos programas independientes y se comparan los momentos articulares (en el marco
local de cada articulación) y la fuerza/momento residual de la raíz.
Salida: P2_DinamicaInversa/resultados/validacion_mujoco.json
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

from nucleo.esqueleto import Esqueleto, PADRES, NJ_CUERPO, NOMBRES, pose_anatomica, raiz  # noqa: E402
from nucleo.antropometria import ModeloInercial  # noqa: E402
from nucleo import dinamica as D  # noqa: E402
from nucleo.rotaciones import matriz_a_eje_angulo  # noqa: E402


def mjcf(esq, mod):
    def cuerpo(j, sang):
        m = max(mod.m[j], 1e-4)
        I = mod.I[j] if mod.m[j] > 0 else np.eye(3) * 1e-7
        c = mod.c[j]
        pos = esq.offsets[j] if j > 0 else np.zeros(3)
        junta = '<freejoint name="raiz"/>' if j == 0 else f'<joint name="j{j}" type="ball" damping="0"/>'
        hijos = "".join(cuerpo(k, sang + "  ") for k in range(NJ_CUERPO) if PADRES[k] == j)
        return (f'{sang}<body name="{NOMBRES[j]}" pos="{pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}">\n'
                f'{sang}  {junta}\n'
                f'{sang}  <inertial pos="{c[0]:.9f} {c[1]:.9f} {c[2]:.9f}" mass="{m:.9f}" '
                f'fullinertia="{I[0,0]:.12f} {I[1,1]:.12f} {I[2,2]:.12f} {I[0,1]:.12f} {I[0,2]:.12f} {I[1,2]:.12f}"/>\n'
                f'{hijos}{sang}</body>\n')
    return ('<mujoco model="eb_cuerpo">\n  <option gravity="0 -9.81 0" timestep="0.001" integrator="RK4"/>\n'
            '  <worldbody>\n' + cuerpo(0, "    ") + '  </worldbody>\n</mujoco>\n')


def mat_a_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def main():
    rng = np.random.default_rng(3)
    esq = Esqueleto(1.78)
    mod = ModeloInercial(esq, 78.0)
    xml = mjcf(esq, mod)
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "cuerpo_eb.xml").write_text(xml, encoding="utf-8")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    # movimiento suave: sentadilla con brazos y giro de tronco, muestreado a 1 kHz
    fs = 1000.0
    T = 2000
    t = np.arange(T) / fs
    s = 0.5 - 0.5 * np.cos(2 * np.pi * 0.7 * t)
    kw = dict(caderaI_flex=100 * s, caderaD_flex=95 * s, rodillaI_flex=110 * s, rodillaD_flex=105 * s,
              tobilloI_flex=25 * s, tobilloD_flex=22 * s, hombroI_flex=80 * s, hombroD_flex=70 * s + 10 * np.sin(3 * t),
              codoI_flex=40 * s, codoD_flex=60 * s, tronco_flex=15 * s, tronco_rot=10 * np.sin(2 * t),
              caderaI_abd=8 + 4 * np.sin(1.3 * t), caderaD_rot=-10 * s, cuello_flex=10 * np.sin(1.7 * t))
    R = pose_anatomica(T, **kw)
    Rr = raiz(T, pitch=30 * s, yaw=15 * np.sin(0.9 * t), roll=5 * np.sin(1.1 * t))
    pt = np.stack([0.05 * np.sin(1.3 * t), 1.0 - 0.35 * s, -0.12 * s], -1)
    G, P = esq.fk(R, Rr, pt)
    # --- MuJoCo: qpos por fotograma, qvel/qacc por diferencias en la variedad ---
    Q = np.zeros((T, m.nq))
    for i in range(T):
        Q[i, :3] = pt[i]
        Q[i, 3:7] = mat_a_quat(Rr[i] @ R[i, 0])
        for j in range(1, NJ_CUERPO):
            adr = m.jnt_qposadr[m.joint(f"j{j}").id]
            Q[i, adr:adr + 4] = mat_a_quat(R[i, j])
    V = np.zeros((T, m.nv))
    for i in range(1, T - 1):
        mujoco.mj_differentiatePos(m, V[i], 2.0 / fs, Q[i - 1], Q[i + 1])
    A = np.zeros((T, m.nv))
    A[1:-1] = (V[2:] - V[:-2]) * fs / 2
    tau_mj = np.zeros((T, m.nv))
    for i in range(2, T - 2):
        d.qpos[:] = Q[i]
        d.qvel[:] = V[i]
        d.qacc[:] = A[i]
        mujoco.mj_inverse(m, d)
        tau_mj[i] = d.qfrc_inverse
    # --- Newton–Euler propio sobre la cinemática del mismo movimiento (sin filtrar) ---
    cin = D.cinematica_segmentos(mod, G, P, fs, fc=None)
    Fj, Mj = D.newton_euler(mod, cin, externas=[])
    errs = {}
    sl = slice(10, T - 10)
    for j in range(1, NJ_CUERPO):
        dof = m.jnt_dofadr[m.joint(f"j{j}").id]
        mj_loc = tau_mj[sl, dof:dof + 3]
        ne_loc = np.einsum("tba,tb->ta", G[sl, j], Mj[sl, j])
        escala = np.abs(mj_loc).max() + 1e-9
        errs[NOMBRES[j]] = dict(max_abs_Nm=float(np.abs(mj_loc - ne_loc).max()), pico_Nm=float(escala),
                                rel=float(np.abs(mj_loc - ne_loc).max() / escala))
    # raíz: fuerza (mundo) y momento (local) = residuo que el entorno debería aportar
    F0_mj = tau_mj[sl, 0:3]
    M0_mj = tau_mj[sl, 3:6]
    F0_ne = Fj[sl, 0]
    M0_ne = np.einsum("tba,tb->ta", G[sl, 0], Mj[sl, 0])
    errs["raiz_fuerza"] = dict(max_abs=float(np.abs(F0_mj - F0_ne).max()), pico=float(np.abs(F0_mj).max()))
    errs["raiz_momento"] = dict(max_abs=float(np.abs(M0_mj - M0_ne).max()), pico=float(np.abs(M0_mj).max()))
    peor = max(v.get("rel", 0) for v in errs.values() if v.get("pico_Nm", 1) > 0.1)
    # --- 2) con fuerzas externas en los pies (la vía de las reacciones del suelo) ---
    Fext = {7: np.array([15.0, 380.0, -25.0]), 8: np.array([-12.0, 410.0, 30.0])}
    ext = []
    tau2 = tau_mj.copy()
    for j, F in Fext.items():
        r = P[:, j] + np.einsum("tab,b->ta", G[:, j], np.array([0.0, -0.05, 0.06]))
        ext.append((j, np.broadcast_to(F, (T, 3)).copy(), r, None))
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        for i in range(2, T - 2):
            d.qpos[:] = Q[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jac(m, d, jacp, jacr, r[i], m.body(NOMBRES[j]).id)
            tau2[i] -= jacp.T @ F                         # la fuerza externa ayuda: las juntas aportan menos
    Fj2, Mj2 = D.newton_euler(mod, cin, externas=ext)
    errs2 = {}
    for j in range(1, NJ_CUERPO):
        dof = m.jnt_dofadr[m.joint(f"j{j}").id]
        a = tau2[sl, dof:dof + 3]
        b = np.einsum("tba,tb->ta", G[sl, j], Mj2[sl, j])
        esc = np.abs(a).max() + 1e-9
        errs2[NOMBRES[j]] = dict(max_abs_Nm=float(np.abs(a - b).max()), pico_Nm=float(esc), rel=float(np.abs(a - b).max() / esc))
    peor2 = max(v["rel"] for v in errs2.values() if v["pico_Nm"] > 0.1)
    for k, v in errs2.items():
        print("con fuerzas externas", f"{k:16s} {v}")
    res = dict(descripcion="Newton-Euler propio vs MuJoCo mj_inverse, mismo cuerpo y movimiento, 1 kHz",
               masa=78.0, estatura=1.78, fotogramas=T, peor_error_relativo_articular=peor,
               peor_error_relativo_con_fuerzas_externas=peor2, detalle=errs, detalle_fuerzas_externas=errs2)
    (AQUI / "resultados" / "validacion_mujoco.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in errs.items():
        print(f"{k:16s} {v}")
    print("peor error relativo articular:", peor, " con fuerzas externas:", peor2)


if __name__ == "__main__":
    main()
