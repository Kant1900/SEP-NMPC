"""Energy-consistent 10-state rigid-cable quadrotor--payload model.

Coordinates: x,y,z in metres with z UP; swing angles alpha, beta are the
x-z/y-z projections used in arXiv:2603.08860, section II.

State: [x,y,z,alpha,beta, vx,vy,vz,alpha_dot,beta_dot].
Control: shaped force u_a in newtons (not motor commands).
Physical force F = u_a - K (p - p_ref) + (m_Q + m_L) g e_z.
"""

from dataclasses import dataclass, field

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class Parameters:
    m_q: float = 1.5
    m_l: float = 0.20
    length: float = 0.50
    g: float = 9.81
    k_pos: tuple[float, float, float] = (2.5, 2.5, 4.0)
    rho: float = 0.08
    epsilon: float = 0.02
    cbf_k1: float = 3.0
    cbf_k2: float = 3.0
    max_swing_deg: float = 65.0
    radius_quad: float = 0.14
    radius_load: float = 0.08
    safety_margin: float = 0.08
    obstacle_radii: tuple[float, float] = (0.33, 0.33)
    obstacle_mode: str = "cylinder"  # 2D infinite-height, or "sphere" (3D)


@dataclass
class Model:
    cfg: Parameters
    x: ca.SX
    u: ca.SX
    p: ca.SX
    xdot: ca.SX
    mass: ca.SX
    coriolis_times_velocity: ca.SX
    gravity: ca.SX
    force: ca.SX
    storage: ca.SX
    storage_dot: ca.SX
    passivity: ca.SX
    cbf: ca.SX
    barrier: ca.SX
    barrier_dot: ca.SX
    barrier_psi1: ca.SX
    p_quad: ca.SX
    p_load: ca.SX
    v_quad: ca.SX
    v_load: ca.SX
    a_quad: ca.SX
    a_load: ca.SX
    fun: ca.Function = field(init=False)
    mass_fun: ca.Function = field(init=False)
    checks: ca.Function = field(init=False)

    def __post_init__(self):
        self.fun = ca.Function("f_rigid_load", [self.x, self.u, self.p], [self.xdot])
        self.mass_fun = ca.Function("inertia", [self.x], [self.mass])
        self.checks = ca.Function(
            "check_constraints", [self.x, self.u, self.p],
            [self.passivity, self.cbf, self.barrier, self.barrier_psi1,
             self.storage, self.storage_dot, self.force, self.p_quad, self.p_load,
             self.a_quad, self.a_load],
            ["x", "u_a", "parameters"],
            ["passivity", "cbf", "h", "psi1", "energy", "energy_dot",
             "force", "quad", "load", "a_quad", "a_load"],
        )


def make_model(cfg: Parameters = Parameters()) -> Model:
    """Derive M, C(q,dq)dq and G from a single Lagrangian.

    This intentionally does not transcribe the article's displayed C matrix:
    the provided list is not sufficient to assert energy consistency. Its
    payload acceleration input Jacobian is also not assumed equal to the UAV's.
    """
    if cfg.obstacle_mode not in ("cylinder", "sphere"):
        raise ValueError("obstacle_mode must be cylinder or sphere")
    x = ca.SX.sym("x", 10)
    u_a = ca.SX.sym("u_a", 3)
    # [target position (3), obstacle_0 center/velocity/acceleration (9),
    #  obstacle_1 center/velocity/acceleration (9)]
    p = ca.SX.sym("parameters", 21)
    q, dq = x[:5], x[5:]
    alpha, beta = q[3], q[4]
    l, ml, mq, g = cfg.length, cfg.m_l, cfg.m_q, cfg.g

    # p_L = p_Q + l*[sin(a)cos(b), sin(b), -cos(a)cos(b)]
    offset = l * ca.vertcat(
        ca.sin(alpha) * ca.cos(beta), ca.sin(beta),
        -ca.cos(alpha) * ca.cos(beta),
    )
    pos_load = q[:3] + offset
    J_load = ca.jacobian(pos_load, q)
    vel_load = J_load @ dq
    mass = mq * ca.diag(ca.vertcat(1, 1, 1, 0, 0)) + ml * (J_load.T @ J_load)
    Tkin = 0.5 * ca.mtimes([dq.T, mass, dq])
    momentum = mass @ dq
    C_times_dq = ca.jacobian(momentum, q) @ dq - ca.gradient(Tkin, q)
    Ugrav = (mq + ml) * g * q[2] - ml * g * l * ca.cos(alpha) * ca.cos(beta)
    gravity = ca.gradient(Ugrav, q)
    K = ca.diag(ca.DM(cfg.k_pos))
    epos = q[:3] - p[:3]
    hover_force = ca.vertcat(0, 0, (mq + ml) * g)
    force = u_a - K @ epos + hover_force
    generalized = ca.vertcat(force, ca.SX.zeros(2))
    ddq = ca.solve(mass, generalized - C_times_dq - gravity)
    xdot = ca.vertcat(dq, ddq)

    # Energy with mg*z removed, which is cancelled by the hover feedforward.
    # The exact port relation for a stationary reference is Vdot = v^T u_a.
    storage = (Tkin + ml * g * l * (1 - ca.cos(alpha) * ca.cos(beta))
               + 0.5 * ca.mtimes([epos.T, K, epos]))
    storage_dot = ca.jacobian(storage, x) @ xdot
    v_quad, v_load = dq[:3], vel_load
    a_quad = ddq[:3]
    a_load = ca.jacobian(vel_load, x) @ xdot

    # <=0 in article, >=0 equivalent for acados / diagnostics.
    passivity = -(ca.dot(u_a, v_quad) + cfg.rho * ca.dot(v_quad, v_quad)
                  + cfg.epsilon * ca.dot(u_a, u_a))

    psi2_all, h_all, hdot_all, psi1_all = [], [], [], []
    for i, obstacle_radius in enumerate(cfg.obstacle_radii):
        offset_p = 3 + i * 9
        obs_pos = p[offset_p:offset_p + 3]
        obs_vel = p[offset_p + 3:offset_p + 6]
        obs_acc = p[offset_p + 6:offset_p + 9]
        for body_pos, body_vel, body_acc, body_radius in (
            (q[:3], v_quad, a_quad, cfg.radius_quad),
            (pos_load, v_load, a_load, cfg.radius_load),
        ):
            r = body_pos - obs_pos
            vr = body_vel - obs_vel
            ar = body_acc - obs_acc
            if cfg.obstacle_mode == "cylinder":
                r, vr, ar = r[:2], vr[:2], ar[:2]
            clearance = obstacle_radius + body_radius + cfg.safety_margin
            h = ca.dot(r, r) - clearance**2
            hdot = 2 * ca.dot(r, vr)
            hddot = 2 * ca.dot(vr, vr) + 2 * ca.dot(r, ar)
            psi1 = hdot + cfg.cbf_k1 * h
            psi2 = hddot + (cfg.cbf_k1 + cfg.cbf_k2) * hdot + cfg.cbf_k1 * cfg.cbf_k2 * h
            h_all.append(h)
            hdot_all.append(hdot)
            psi1_all.append(psi1)
            psi2_all.append(psi2)

    return Model(
        cfg, x, u_a, p, xdot, mass, C_times_dq, gravity, force,
        storage, storage_dot, passivity, ca.vertcat(*psi2_all),
        ca.vertcat(*h_all), ca.vertcat(*hdot_all), ca.vertcat(*psi1_all),
        q[:3], pos_load, v_quad, v_load, a_quad, a_load,
    )


def rk4_step(model: Model, x: np.ndarray, u: np.ndarray, parameters: np.ndarray,
             dt: float, n_substeps: int = 2) -> np.ndarray:
    """Independent plant integration; zero-order held shaped control input."""
    state = np.asarray(x, dtype=float).reshape(10).copy()
    h = dt / n_substeps
    f = model.fun
    for _ in range(n_substeps):
        k1 = np.asarray(f(state, u, parameters)).ravel()
        k2 = np.asarray(f(state + 0.5 * h * k1, u, parameters)).ravel()
        k3 = np.asarray(f(state + 0.5 * h * k2, u, parameters)).ravel()
        k4 = np.asarray(f(state + h * k3, u, parameters)).ravel()
        state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return state
