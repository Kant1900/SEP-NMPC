"""Run with: python -m unittest discover -s tests -v"""

import unittest

import casadi as ca
import numpy as np

from sep_nmpc.model import Parameters, make_model, rk4_step
from run_sim import get_scene, parameters_at


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Parameters()
        cls.model = make_model(cls.cfg)
        cls.scene = get_scene("static")

    def test_equilibrium_and_hover_force(self):
        x = np.array([0., 0., 1.5, 0., 0., 0., 0., 0., 0., 0.])
        p = parameters_at(get_scene("hover"), 0.).copy()
        p[:3] = x[:3]
        deriv = np.asarray(self.model.fun(x, np.zeros(3), p)).ravel()
        np.testing.assert_allclose(deriv, 0., atol=1e-11)
        force = np.asarray(self.model.checks(x, np.zeros(3), p)[6]).ravel()
        np.testing.assert_allclose(force, [0, 0, (self.cfg.m_q+self.cfg.m_l)*self.cfg.g])

    def test_mass_positive_definite(self):
        for a, b in [(0., 0.), (0.2, -0.3), (1.0, 0.6)]:
            x = np.array([0., 0., 1.5, a, b, 0., 0., 0., 0., 0.])
            mass = np.asarray(self.model.mass_fun(x))
            np.testing.assert_allclose(mass, mass.T, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(mass)[0], 0.)

    def test_exact_passivity_power_identity(self):
        # Formula is for a stationary target, with the hover mg compensation.
        x = np.array([0.4, -0.6, 1.2, 0.23, -0.16, 0.8, -0.3, .13, 0.5, -0.2])
        p = parameters_at(self.scene, 0.)
        u = np.array([-1.2, 0.4, -0.8])
        out = self.model.checks(x, u, p)
        storage_dot = float(out[5])
        self.assertAlmostEqual(storage_dot, np.dot(x[5:8], u), places=9)

    def test_payload_and_quad_control_maps_differ(self):
        # For the rigid cable, using a single shared G_v for both is incorrect.
        x = np.array([0.1, 0.2, 1.5, 0.14, -0.28, 0, 0, 0, 0, 0])
        p = parameters_at(self.scene, 0.)
        g_q = np.column_stack([
            np.asarray(self.model.checks(x, np.eye(3)[i], p)[9]).ravel()
            - np.asarray(self.model.checks(x, np.zeros(3), p)[9]).ravel()
            for i in range(3)
        ])
        g_l = np.column_stack([
            np.asarray(self.model.checks(x, np.eye(3)[i], p)[10]).ravel()
            - np.asarray(self.model.checks(x, np.zeros(3), p)[10]).ravel()
            for i in range(3)
        ])
        self.assertGreater(np.linalg.norm(g_q - g_l), 0.1)

    def test_cbf_affine_in_input(self):
        x = np.array([0.15, -0.55, 1.5, 0.18, 0.1, 0.4, 0.2, 0., -0.1, 0.05])
        p = parameters_at(self.scene, 0.)
        u = np.array([1.4, -0.7, 0.1])
        c0 = np.asarray(self.model.checks(x, np.zeros(3), p)[1]).ravel()
        cp = np.asarray(self.model.checks(x, u, p)[1]).ravel()
        cm = np.asarray(self.model.checks(x, -u, p)[1]).ravel()
        np.testing.assert_allclose(cp + cm, 2*c0, atol=1e-9)

    def test_zero_input_rk4_matches_equilibrium(self):
        x = np.array([0., 0., 1.5, 0., 0., 0., 0., 0., 0., 0.])
        p = parameters_at(get_scene("hover"), 0.).copy()
        p[:3] = x[:3]
        step = rk4_step(self.model, x, np.zeros(3), p, 0.05)
        np.testing.assert_allclose(step, x, atol=1e-11)

    def test_scene_initial_cbfs_safe(self):
        for scene_name in ("hover", "static", "dynamic"):
            scene = get_scene(scene_name)
            out = self.model.checks(scene.x0, np.zeros(3), parameters_at(scene, 0.))
            self.assertGreaterEqual(np.min(np.asarray(out[2])), 0.)
            self.assertGreaterEqual(np.min(np.asarray(out[3])), 0.)


if __name__ == "__main__":
    unittest.main()
