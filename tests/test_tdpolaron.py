import unittest

import numpy as np

from Methods.TDPolaronNZ2 import (
    DiscreteHarmonicBath,
    NUMBA_AVAILABLE,
    SIGMA_X,
    SIGMA_Z,
    propagate_td_polaron_nz2,
)
from Models.SpinBoson_Benchmarks import (
    BENCHMARKS,
    _legacy_ohmic_discretization,
    get_benchmark,
)


class TDPolaronTests(unittest.TestCase):
    def test_registry_contains_legacy_models(self):
        self.assertEqual(sorted(BENCHMARKS), list(range(3, 20)))
        self.assertEqual(get_benchmark(8).bias, 2.0)
        self.assertEqual(get_benchmark(8).tunneling, 1.0)


    def test_legacy_bath_mapping_reproduces_bare_correlation(self):
        spec = get_benchmark(8).with_overrides(n_modes=40)
        coordinate_coupling, omega = _legacy_ohmic_discretization(
            cutoff=spec.cutoff, alpha=spec.strength, n_modes=spec.n_modes
        )
        bath = spec.build_bath()
        time = 0.73
        coth = 1.0 / np.tanh(0.5 * spec.beta * omega)
        legacy = 0.25 * np.sum(
            coordinate_coupling**2
            / (2.0 * omega)
            * (coth * np.cos(omega * time) - 1.0j * np.sin(omega * time))
        )
        mapped = np.sum(
            bath.coupling**2
            * (coth * np.cos(omega * time) - 1.0j * np.sin(omega * time))
        )
        self.assertLess(abs(legacy - mapped), 1e-14)

    def test_zero_coupling_reproduces_isolated_two_level_system(self):
        bath = DiscreteHarmonicBath(
            omega=np.array([1.0]), coupling=np.array([0.0]), beta=1.5
        )
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        bias = 1.2
        tunneling = 0.7
        trajectory = propagate_td_polaron_nz2(
            bias=bias,
            tunneling=tunneling,
            bath=bath,
            rho0=rho0,
            t_max=1.0,
            dt=0.01,
            method="td",
        )

        H = 0.5 * bias * SIGMA_Z + tunneling * SIGMA_X
        omega_eff = np.sqrt((0.5 * bias) ** 2 + tunneling**2)
        exact = []
        eye = np.eye(2, dtype=complex)
        for time in trajectory.times:
            U = (
                np.cos(omega_eff * time) * eye
                - 1.0j * np.sin(omega_eff * time) / omega_eff * H
            )
            exact.append(U @ rho0 @ U.conj().T)
        exact = np.asarray(exact)
        self.assertLess(np.max(np.abs(trajectory.rho_s - exact)), 2e-12)

    def test_initial_connected_correlation_is_zero_indirectly(self):
        # At t=0 the TD displacement is the identity, so the interaction-picture
        # density matrix initially has zero derivative.  The first finite-time
        # change must therefore be second order in dt.
        bath = DiscreteHarmonicBath(
            omega=np.array([0.8, 2.0]),
            coupling=np.array([0.2, 0.1]),
            beta=2.0,
        )
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        dt = 1e-3
        trajectory = propagate_td_polaron_nz2(
            bias=0.0,
            tunneling=0.5,
            bath=bath,
            rho0=rho0,
            t_max=2 * dt,
            dt=dt,
            method="td",
        )
        self.assertAlmostEqual(trajectory.dressing[0], 1.0, places=15)
        first_change = np.max(
            np.abs(trajectory.rho_interaction[1] - trajectory.rho_interaction[0])
        )
        self.assertLess(first_change, 1e-9)

    def test_short_run_preserves_trace(self):
        bath = DiscreteHarmonicBath(
            omega=np.array([0.4, 1.0, 2.2]),
            coupling=np.array([0.08, 0.12, 0.05]),
            beta=1.0,
        )
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        trajectory = propagate_td_polaron_nz2(
            bias=0.4,
            tunneling=0.6,
            bath=bath,
            rho0=rho0,
            t_max=0.5,
            dt=0.01,
            method="td",
        )
        self.assertLess(np.max(np.abs(trajectory.trace - 1.0)), 1e-10)
        self.assertLess(
            np.max(np.abs(trajectory.rho_s - trajectory.rho_s.conj().transpose(0, 2, 1))),
            1e-12,
        )



    def test_static_underflow_overlap_remains_finite(self):
        # Model 5 has an infrared-heavy discretization for which kappa
        # underflows to zero.  The same-channel static correlation must be
        # evaluated in log form rather than as 0 * infinity.
        spec = get_benchmark(5).with_overrides(t_max=0.05, dt=0.01)
        bath = spec.build_bath()
        self.assertEqual(bath.kappa(), 0.0)
        result = propagate_td_polaron_nz2(
            bias=spec.bias,
            tunneling=spec.tunneling,
            bath=bath,
            rho0=spec.rho0,
            t_max=spec.t_max,
            dt=spec.dt,
            method="static",
            backend="numpy",
        )
        self.assertTrue(np.all(np.isfinite(result.rho_s)))

    @unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
    def test_numba_matches_numpy_backend(self):
        bath = DiscreteHarmonicBath(
            omega=np.array([0.4, 1.0, 2.2]),
            coupling=np.array([0.08, 0.12, 0.05]),
            beta=1.0,
        )
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        kwargs = dict(
            bias=0.4,
            tunneling=0.6,
            bath=bath,
            rho0=rho0,
            t_max=0.5,
            dt=0.01,
        )
        for method in ("td", "static"):
            numpy_result = propagate_td_polaron_nz2(
                method=method, backend="numpy", **kwargs
            )
            numba_result = propagate_td_polaron_nz2(
                method=method, backend="numba", **kwargs
            )
            self.assertLess(
                np.max(np.abs(numpy_result.rho_s - numba_result.rho_s)), 1e-12
            )

    def test_static_zero_coupling_matches_td(self):
        bath = DiscreteHarmonicBath(
            omega=np.array([0.7]), coupling=np.array([0.0]), beta=2.0
        )
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        kwargs = dict(
            bias=0.2,
            tunneling=0.4,
            bath=bath,
            rho0=rho0,
            t_max=0.4,
            dt=0.01,
        )
        td = propagate_td_polaron_nz2(method="td", **kwargs)
        static = propagate_td_polaron_nz2(method="static", **kwargs)
        self.assertLess(np.max(np.abs(td.rho_s - static.rho_s)), 2e-12)


if __name__ == "__main__":
    unittest.main()
