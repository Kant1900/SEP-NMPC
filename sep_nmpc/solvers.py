"""Optimal-control formulations sharing the same CasADi physical model.

Acados is the intended fast solver. IPOPT is an explicitly slower, convenient
fallback for verifying the mathematics on machines without acados.
"""

from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

from .model import Model


Q_STATE = np.array([18., 18., 24., 5., 5., 3., 3., 4., 1.5, 1.5])
R_INPUT = np.array([0.08, 0.08, 0.1])
Q_TERMINAL = 3. * Q_STATE
FORCE_LOWER = np.array([-8., -8., 5.])
FORCE_UPPER = np.array([8., 8., 32.])
INPUT_LOWER = np.array([-18., -18., -18.])
INPUT_UPPER = np.array([18., 18., 18.])


@dataclass(frozen=True)
class OcpSettings:
    horizon: float = 2.0
    nodes: int = 40
    rti: bool = True

    @property
    def dt(self) -> float:
        return self.horizon / self.nodes


def parameterized_constraints(m: Model) -> ca.SX:
    """Passivity, HOCBF, sampled-set guards and physical-force saturation."""
    return ca.vertcat(m.passivity, m.cbf, m.barrier, m.barrier_psi1, m.force)


def x_reference(ref_position):
    return ca.vertcat(ref_position, ca.DM.zeros(7, 1))


class AcadosController:
    """SQP-RTI MPC with RK4 (ERK 4 stages, one step per shooting interval)."""

    def __init__(self, m: Model, settings: OcpSettings, build_dir: Path):
        try:
            from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        except ImportError as exc:
            raise RuntimeError(
                "acados_template is missing; install acados and its Python interface "
                "(see README), or run --solver ipopt for a numerical smoke test."
            ) from exc

        self.m = m
        self.settings = settings
        build_dir = Path(build_dir).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        model = AcadosModel()
        model.name = "sep_nmpc_rigid_load"
        model.x = m.x
        model.u = m.u
        model.p = m.p
        model.xdot = ca.SX.sym("xdot", 10)
        model.f_expl_expr = m.xdot
        model.f_impl_expr = model.xdot - m.xdot
        model.cost_y_expr = ca.vertcat(m.x, m.u)
        model.cost_y_expr_e = m.x
        model.con_h_expr = parameterized_constraints(m)
        # Stage 0 is not automatically guaranteed to inherit path constraints.
        model.con_h_expr_0 = model.con_h_expr
        model.con_h_expr_e = ca.vertcat(m.barrier, m.barrier_psi1)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = settings.nodes
        ocp.solver_options.tf = settings.horizon
        ocp.parameter_values = np.zeros(21)
        ocp.code_export_directory = str(build_dir / "c_generated_code")

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.diag(np.r_[Q_STATE, R_INPUT])
        ocp.cost.W_e = np.diag(Q_TERMINAL)
        ocp.cost.yref = np.zeros(13)
        ocp.cost.yref_e = np.zeros(10)

        # Constraints: pass >= 0, cbf(4) >=0, h(4)>=0, psi1(4)>=0,
        # then physical force saturation on the final three entries.
        nl_lower = np.r_[np.zeros(13), FORCE_LOWER]
        nl_upper = np.r_[np.full(13, 1e9), FORCE_UPPER]
        ocp.constraints.lh = nl_lower
        ocp.constraints.uh = nl_upper
        ocp.constraints.lh_0 = nl_lower.copy()
        ocp.constraints.uh_0 = nl_upper.copy()
        ocp.constraints.lh_e = np.zeros(8)
        ocp.constraints.uh_e = np.full(8, 1e9)
        ocp.constraints.idxbu = np.arange(3)
        ocp.constraints.lbu = INPUT_LOWER
        ocp.constraints.ubu = INPUT_UPPER
        ocp.constraints.idxbx = np.array([3, 4], dtype=int)
        angle = np.deg2rad(m.cfg.max_swing_deg)
        ocp.constraints.lbx = np.array([-angle, -angle])
        ocp.constraints.ubx = np.array([angle, angle])
        ocp.constraints.x0 = np.zeros(10)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.nlp_solver_type = "SQP_RTI" if settings.rti else "SQP"
        ocp.solver_options.print_level = 0
        self.solver = AcadosOcpSolver(
            ocp, json_file=str(build_dir / "acados_ocp.json")
        )
        self.prev_x = None
        self.prev_u = None

    def solve(self, x0: np.ndarray, p_stages: np.ndarray) -> tuple[np.ndarray, int]:
        N = self.settings.nodes
        if p_stages.shape != (N + 1, 21):
            raise ValueError(f"Expected {(N+1,21)} parameters; got {p_stages.shape}")
        for k in range(N + 1):
            self.solver.set(k, "p", p_stages[k])
            yref = np.r_[p_stages[k, :3], np.zeros(7)]
            if k < N:
                self.solver.cost_set(k, "yref", np.r_[yref, np.zeros(3)])
            else:
                self.solver.cost_set(k, "yref", yref)

        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)
        if self.prev_u is None:
            for k in range(N + 1):
                self.solver.set(k, "x", x0)
            for k in range(N):
                self.solver.set(k, "u", np.zeros(3))
        else:
            for k in range(N):
                self.solver.set(k, "x", self.prev_x[min(k+1, N)])
                self.solver.set(k, "u", self.prev_u[min(k+1, N-1)])
            self.solver.set(N, "x", self.prev_x[N])
        status = self.solver.solve()
        if status != 0:
            raise RuntimeError(f"acados returned status={status}")
        self.prev_u = np.array([self.solver.get(k, "u") for k in range(N)])
        self.prev_x = np.array([self.solver.get(k, "x") for k in range(N + 1)])
        return self.prev_u[0].copy(), status


class IpoptController:
    """Slower multiple-shooting mathematical fallback, not an RTI benchmark."""

    def __init__(self, m: Model, settings: OcpSettings):
        self.m = m
        self.settings = settings
        N, dt = settings.nodes, settings.dt
        opti = ca.Opti()
        X = opti.variable(10, N + 1)
        U = opti.variable(3, N)
        x0 = opti.parameter(10)
        params = opti.parameter(21, N + 1)
        constraints = ca.Function(
            "path_constraints", [m.x, m.u, m.p], [parameterized_constraints(m)]
        )
        obj = 0
        opti.subject_to(X[:, 0] == x0)
        max_angle = np.deg2rad(m.cfg.max_swing_deg)
        for k in range(N):
            state = X[:, k]
            u = U[:, k]
            param = params[:, k]
            error = state - ca.vertcat(param[:3], ca.MX.zeros(7, 1))
            obj += dt * (ca.dot(ca.DM(Q_STATE) * error, error)
                         + ca.dot(ca.DM(R_INPUT) * u, u))
            opti.subject_to(opti.bounded(INPUT_LOWER, u, INPUT_UPPER))
            opti.subject_to(opti.bounded(-max_angle, state[3:5], max_angle))
            g = constraints(state, u, param)
            opti.subject_to(g[:13] >= 0)
            opti.subject_to(opti.bounded(FORCE_LOWER, g[13:16], FORCE_UPPER))
            # Exactly RK4 over one shooting interval for the prediction model.
            k1 = m.fun(state, u, param)
            k2 = m.fun(state + 0.5 * dt * k1, u, param)
            k3 = m.fun(state + 0.5 * dt * k2, u, param)
            k4 = m.fun(state + dt * k3, u, param)
            xn = state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k+1] == xn)
        terminal = X[:, N]
        e_terminal = terminal - ca.vertcat(params[:3, N], ca.MX.zeros(7, 1))
        obj += ca.dot(ca.DM(Q_TERMINAL) * e_terminal, e_terminal)
        opti.subject_to(opti.bounded(-max_angle, terminal[3:5], max_angle))
        barriers = ca.Function("end_barriers", [m.x, m.p],
                               [ca.vertcat(m.barrier, m.barrier_psi1)])
        opti.subject_to(barriers(terminal, params[:, N]) >= 0)
        opti.minimize(0.5 * obj)
        opti.solver("ipopt", {"expand": False, "print_time": False}, {
            "print_level": 0, "sb": "yes", "max_iter": 450,
            "tol": 1e-5, "acceptable_tol": 5e-4,
        })
        self.opti, self.X, self.U = opti, X, U
        self.x0, self.params = x0, params
        self.prev_x = None
        self.prev_u = None

    def solve(self, x0: np.ndarray, p_stages: np.ndarray) -> tuple[np.ndarray, int]:
        opti, N = self.opti, self.settings.nodes
        if p_stages.shape != (N+1, 21):
            raise ValueError("Unexpected parameter array shape")
        opti.set_value(self.x0, x0)
        opti.set_value(self.params, p_stages.T)
        if self.prev_x is None:
            opti.set_initial(self.X, np.tile(x0[:, None], (1, N + 1)))
            opti.set_initial(self.U, np.zeros((3, N)))
        else:
            opti.set_initial(self.X, np.c_[self.prev_x[1:].T, self.prev_x[-1]])
            opti.set_initial(self.U, np.c_[self.prev_u[1:].T, self.prev_u[-1]])
        try:
            solution = opti.solve()
        except RuntimeError as exc:
            raise RuntimeError(f"IPOPT did not converge: {exc}") from exc
        self.prev_x = np.asarray(solution.value(self.X)).T
        self.prev_u = np.asarray(solution.value(self.U)).T
        return self.prev_u[0].copy(), 0
