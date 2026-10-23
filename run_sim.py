#!/usr/bin/env python3
"""Run stage-one SEP-NMPC closed-loop simulation.

Examples:
    python run_sim.py --solver ipopt --scenario hover --nodes 8 --steps 12
    python run_sim.py --solver acados --scenario static --steps 120
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from sep_nmpc.model import Parameters, make_model, rk4_step
from sep_nmpc.solvers import OcpSettings, AcadosController, IpoptController


@dataclass(frozen=True)
class Scene:
    x0: np.ndarray
    target: np.ndarray
    centers: tuple[tuple[float, float, float], ...]
    velocities: tuple[tuple[float, float, float], ...]


def get_scene(name: str) -> Scene:
    if name == "hover":
        return Scene(
            np.array([0., 0., 1.5, 0., 0., 0., 0., 0., 0., 0.]),
            np.array([0.8, 0.25, 1.5]),
            ((100., 100., 1.5), (105., 105., 1.5)),
            ((0., 0., 0.), (0., 0., 0.)),
        )
    if name == "static":
        return Scene(
            np.array([0., -0.9, 1.5, 0., 0., 0., 0., 0., 0., 0.]),
            np.array([2.6, 0.7, 1.5]),
            ((1.1, -0.05, 1.5), (1.85, 0.55, 1.5)),
            ((0., 0., 0.), (0., 0., 0.)),
        )
    if name == "dynamic":
        return Scene(
            np.array([0., -0.7, 1.5, 0., 0., 0., 0., 0., 0., 0.]),
            np.array([2.7, 0.5, 1.5]),
            ((1.1, 0.2, 1.5), (1.8, -1.5, 1.5)),
            ((0., 0., 0.), (0., 0.1, 0.)),
        )
    raise ValueError(name)


def parameters_at(scene: Scene, t: float) -> np.ndarray:
    """Moving obstacles use constant velocity and zero acceleration."""
    p = list(scene.target)
    for center, velocity in zip(scene.centers, scene.velocities):
        position = np.asarray(center) + t * np.asarray(velocity)
        p.extend(position)
        p.extend(velocity)
        p.extend((0., 0., 0.))
    return np.asarray(p, dtype=float)


def plot_results(out: Path, scenario: str, scene: Scene, cfg: Parameters,
                 t: np.ndarray, states: np.ndarray, controls: np.ndarray,
                 passivity: np.ndarray, cbfs: np.ndarray,
                 distances: np.ndarray, solve_ms: np.ndarray) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(states[:, 0], states[:, 1], color="tab:blue", label="quadrotor")
    # Compute the load x/y directly from the modeled cable geometry.
    alpha, beta = states[:, 3], states[:, 4]
    lx = states[:, 0] + cfg.length * np.sin(alpha) * np.cos(beta)
    ly = states[:, 1] + cfg.length * np.sin(beta)
    ax.plot(lx, ly, color="tab:orange", linestyle="--", label="payload")
    ax.scatter([states[0, 0]], [states[0, 1]], marker="o", c="green", label="start")
    ax.scatter([scene.target[0]], [scene.target[1]], marker="*", c="red", s=90, label="goal")
    for i, (center, velocity) in enumerate(zip(scene.centers, scene.velocities)):
        if np.linalg.norm(center[:2]) > 50:
            continue
        for ti, alpha_c in ((0., 0.2), (t[-1], 0.08)):
            xy = np.asarray(center[:2]) + ti * np.asarray(velocity[:2])
            ax.add_patch(Circle(xy, cfg.obstacle_radii[i] + cfg.radius_quad + cfg.safety_margin,
                                color="crimson", alpha=alpha_c))
        xy0 = np.asarray(center[:2])
        xy1 = xy0 + t[-1] * np.asarray(velocity[:2])
        ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], ":", color="crimson")
    ax.set_title(f"Top view – {scenario}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(t, states[:, 0], label="x")
    ax.plot(t, states[:, 1], label="y")
    ax.plot(t, states[:, 2], label="z")
    ax.axhline(scene.target[0], c="C0", alpha=0.3, linestyle="--")
    ax.axhline(scene.target[1], c="C1", alpha=0.3, linestyle="--")
    ax.axhline(scene.target[2], c="C2", alpha=0.3, linestyle="--")
    ax.set_title("Position [m]")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axs[1, 0]
    ax.plot(t, np.rad2deg(states[:, 3]), label="alpha")
    ax.plot(t, np.rad2deg(states[:, 4]), label="beta")
    ax.set_title("Payload swing [degrees]")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axs[1, 1]
    if len(passivity):
        tx = t[:-1]
        ax.plot(tx, passivity, label="passivity margin")
        ax.plot(tx, np.min(cbfs, axis=1), label="min HOCBF psi2")
        ax.axhline(0., c="k", lw=0.8)
    ax.set_title("Constraint margins (>= 0 is feasible)")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out / "trajectories.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    if len(distances):
        for j, label in enumerate(("Q/obs0", "L/obs0", "Q/obs1", "L/obs1")):
            ax.plot(t[:-1], distances[:, j], label=label)
    ax.axhline(0., c="k", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("clearance distance [m] (actual distance - required)")
    ax.set_title("Dual-body obstacle clearance")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.savefig(out / "clearances.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("acados", "ipopt"), default="acados")
    parser.add_argument("--scenario", choices=("hover", "static", "dynamic"), default="hover")
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.05, help="plant/control timestep (s)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--full-sqp", action="store_true", help="acados SQP instead of SQP_RTI")
    parser.add_argument("--geometry", choices=("cylinder", "sphere"), default="cylinder")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if min(args.nodes, args.steps, args.dt, args.horizon) <= 0:
        parser.error("--nodes, --steps, --dt, --horizon must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = Parameters(obstacle_mode=args.geometry)
    m = make_model(cfg)
    settings = OcpSettings(args.horizon, args.nodes, not args.full_sqp)
    controller = (AcadosController(m, settings, output / "acados_build")
                  if args.solver == "acados" else IpoptController(m, settings))
    scene = get_scene(args.scenario)
    state = scene.x0.copy()
    states = [state.copy()]
    controls, passivity, cbfs, hs, distances, solve_ms = [], [], [], [], [], []
    failed = False

    initial_checks = m.checks(state, np.zeros(3), parameters_at(scene, 0.))
    initial_h = np.asarray(initial_checks[2]).ravel()
    initial_psi1 = np.asarray(initial_checks[3]).ravel()
    if np.min(initial_h) < 0 or np.min(initial_psi1) < 0:
        raise RuntimeError("Initial CBF safe set violated (requires h >= 0 and psi1 >= 0)")

    for step in range(args.steps):
        now = step * args.dt
        stage_parameters = np.array([
            parameters_at(scene, now + k * settings.dt) for k in range(settings.nodes+1)
        ])
        tic = time.perf_counter()
        try:
            u, _ = controller.solve(state, stage_parameters)
        except RuntimeError as exc:
            print(f"Solver stopped at k={step}, t={now:.3f}s: {exc}")
            failed = True
            break
        solve_ms.append((time.perf_counter() - tic) * 1e3)
        check = m.checks(state, u, stage_parameters[0])
        pass_margin = float(check[0])
        cbf_margin = np.asarray(check[1]).ravel()
        h_values = np.asarray(check[2]).ravel()
        # h=r^2-d^2; compute actual signed distance in metres below.
        d_values = []
        for j, h in enumerate(h_values):
            obstacle = j // 2
            body_radius = cfg.radius_quad if j % 2 == 0 else cfg.radius_load
            dmin = cfg.obstacle_radii[obstacle] + body_radius + cfg.safety_margin
            d_values.append(np.sqrt(max(0., h + dmin*dmin)) - dmin)
        if min(pass_margin, np.min(cbf_margin), np.min(h_values)) < -1e-3:
            print(f"WARNING: sampled constraint violation at step {step}: "
                  f"pass={pass_margin:.2e}, min_cb={min(cbf_margin):.2e}, "
                  f"min_h={min(h_values):.2e}")
        controls.append(u)
        passivity.append(pass_margin)
        cbfs.append(cbf_margin)
        hs.append(h_values)
        distances.append(d_values)
        # Integrate the continuous plant with held control using independent RK4.
        state = rk4_step(m, state, u, stage_parameters[0], args.dt, n_substeps=4)
        states.append(state.copy())
        if (step + 1) % max(1, args.steps // 10) == 0:
            print(f"step={step+1}/{args.steps}, t={now+args.dt:.2f}s, "
                  f"position={np.round(state[:3],3)}, "
                  f"solve={solve_ms[-1]:.1f}ms")

    n = len(controls)
    t = np.arange(n + 1) * args.dt
    states = np.asarray(states)
    controls = np.asarray(controls).reshape(n, 3)
    passivity = np.asarray(passivity)
    cbfs = np.asarray(cbfs).reshape(n, 4)
    hs = np.asarray(hs).reshape(n, 4)
    distances = np.asarray(distances).reshape(n, 4)
    solve_ms = np.asarray(solve_ms)
    np.savez_compressed(
        output / "data.npz", t=t, states=states, controls=controls,
        passivity=passivity, cbf=cbfs, h=hs, clearance=distances,
        solve_ms=solve_ms, target=scene.target, failed=failed,
    )
    plot_results(output, args.scenario, scene, cfg, t, states, controls,
                 passivity, cbfs, distances, solve_ms)
    if n:
        print(f"\nCompleted {n}/{args.steps} steps; solver={'FAILED' if failed else 'OK'}")
        print(f"final position error [m]: {np.linalg.norm(state[:3]-scene.target):.4f}")
        print(f"max swing angle [deg]: {np.max(abs(np.rad2deg(states[:,3:5]))):.3f}")
        print(f"min passivity margin: {np.min(passivity):.3e}")
        print(f"min HOCBF margin: {np.min(cbfs):.3e}")
        print(f"min sampled clearance [m]: {np.min(distances):.4f}")
        print(f"mean solver time [ms]: {np.mean(solve_ms):.2f}")
    print(f"Artifacts: {output}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
