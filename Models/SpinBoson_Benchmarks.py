"""Classic constant-bias spin-boson benchmarks used by this repository.

The parameter sets are the models already listed in ``SpinBoson_Ohmic.py``
and ``SpinBoson_Debye.py`` (original numbering from J. Chem. Phys. 151,
024105 (2019)).  This module only repackages them so that the time-dependent
polaron NZ2 solver can run any model from a command-line model number.

Legacy convention
-----------------
The existing repository writes

    H_S = epsilon sigma_z + Delta sigma_x.

The TD-polaron paper writes

    H_S = (bias/2) sigma_z + V sigma_x.

Therefore the adapter uses ``bias = 2*epsilon`` and ``V = Delta``.

The legacy bath files construct coordinate couplings c_k and evaluate the
bare correlation function with c_k^2/(8 omega_k).  Hence the equivalent
bosonic coupling used by the TD-polaron equations is

    g_k = c_k / sqrt(8 omega_k).

This preserves the discretized bath convention already used in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional

import numpy as np

from Methods.TDPolaronNZ2 import DiscreteHarmonicBath

BathFamily = Literal["ohmic", "debye"]


@dataclass(frozen=True)
class SpinBosonBenchmark:
    model_id: int
    family: BathFamily
    epsilon: float
    delta: float
    beta: float
    cutoff: float
    strength: float
    n_modes: int
    t_max: float
    dt: float = 0.0025
    nskip: int = 10
    description: str = ""

    @property
    def bias(self) -> float:
        return 2.0 * self.epsilon

    @property
    def tunneling(self) -> float:
        return self.delta

    @property
    def rho0(self) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

    def with_overrides(
        self,
        *,
        t_max: Optional[float] = None,
        dt: Optional[float] = None,
        n_modes: Optional[int] = None,
        nskip: Optional[int] = None,
    ) -> "SpinBosonBenchmark":
        return replace(
            self,
            t_max=self.t_max if t_max is None else float(t_max),
            dt=self.dt if dt is None else float(dt),
            n_modes=self.n_modes if n_modes is None else int(n_modes),
            nskip=self.nskip if nskip is None else int(nskip),
        )

    def build_bath(self) -> DiscreteHarmonicBath:
        if self.family == "ohmic":
            coordinate_coupling, omega = _legacy_ohmic_discretization(
                cutoff=self.cutoff,
                alpha=self.strength,
                n_modes=self.n_modes,
            )
            strength_name = f"alpha={self.strength:g}"
        else:
            coordinate_coupling, omega = _legacy_debye_discretization(
                reorganization_energy=self.strength,
                cutoff=self.cutoff,
                n_modes=self.n_modes,
            )
            strength_name = f"lambda={self.strength:g}"

        coupling = coordinate_coupling / np.sqrt(8.0 * omega)
        return DiscreteHarmonicBath(
            omega=omega,
            coupling=coupling,
            beta=self.beta,
            label=(
                f"legacy {self.family} model {self.model_id}; "
                f"omega_c={self.cutoff:g}, {strength_name}, N={self.n_modes}"
            ),
        )


def _legacy_ohmic_discretization(
    *, cutoff: float, alpha: float, n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    if cutoff <= 0.0 or alpha < 0.0 or n_modes <= 0:
        raise ValueError("invalid Ohmic bath parameters")
    index = np.arange(1, n_modes + 1, dtype=float)
    u = index / (n_modes + 1.0)
    omega = -cutoff * np.log1p(-u)
    coordinate_coupling = np.sqrt(alpha * cutoff / (n_modes + 1.0)) * omega
    return coordinate_coupling, omega


def _legacy_debye_discretization(
    *, reorganization_energy: float, cutoff: float, n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    if cutoff <= 0.0 or reorganization_energy < 0.0 or n_modes <= 0:
        raise ValueError("invalid Debye bath parameters")
    index = np.arange(1, n_modes + 1, dtype=float)
    u = index / (n_modes + 1.0)
    omega = cutoff * np.tan(0.5 * np.pi * (1.0 - u))
    coordinate_coupling = (
        np.sqrt(2.0 * reorganization_energy / (n_modes + 1.0)) * omega
    )
    return coordinate_coupling, omega


BENCHMARKS: Dict[int, SpinBosonBenchmark] = {
    # Ohmic models 3-11 from Models/SpinBoson_Ohmic.py
    3: SpinBosonBenchmark(3, "ohmic", 0, 1, 0.25, 5, 0.02, 300, 12),
    4: SpinBosonBenchmark(4, "ohmic", 0, 1, 0.25, 1, 0.10, 300, 12),
    5: SpinBosonBenchmark(5, "ohmic", 0, 1, 0.25, 0.25, 0.40, 300, 12),
    6: SpinBosonBenchmark(6, "ohmic", 1, 1, 0.25, 1, 0.40, 300, 12),
    7: SpinBosonBenchmark(7, "ohmic", 1, 1, 5, 2, 0.40, 300, 10),
    8: SpinBosonBenchmark(8, "ohmic", 1, 1, 5, 2.5, 0.10, 300, 15),
    9: SpinBosonBenchmark(9, "ohmic", 1, 1, 5, 2.5, 0.20, 300, 15),
    10: SpinBosonBenchmark(10, "ohmic", 1, 1, 5, 2.5, 0.40, 300, 15),
    11: SpinBosonBenchmark(11, "ohmic", 1, 1, 10, 2.5, 0.20, 300, 15),
    # Debye models 12-19 from Models/SpinBoson_Debye.py
    12: SpinBosonBenchmark(12, "debye", 0, 1, 0.5, 0.25, 0.025, 500, 40),
    13: SpinBosonBenchmark(13, "debye", 0, 1, 0.5, 0.25, 0.25, 500, 16),
    14: SpinBosonBenchmark(14, "debye", 0, 1, 5, 0.25, 0.25, 500, 30),
    15: SpinBosonBenchmark(15, "debye", 0, 1, 0.5, 5, 0.25, 1000, 12),
    16: SpinBosonBenchmark(16, "debye", 0, 1, 50, 5, 0.25, 500, 30),
    17: SpinBosonBenchmark(17, "debye", 1, 1, 0.5, 0.25, 0.25, 500, 30),
    18: SpinBosonBenchmark(18, "debye", 1, 1, 0.5, 5, 0.25, 500, 12),
    19: SpinBosonBenchmark(19, "debye", 1, 1, 50, 5, 0.25, 1000, 30),
}


def get_benchmark(model_id: int) -> SpinBosonBenchmark:
    try:
        return BENCHMARKS[int(model_id)]
    except (KeyError, ValueError) as exc:
        valid = ", ".join(str(key) for key in sorted(BENCHMARKS))
        raise KeyError(f"unknown model {model_id}; available models: {valid}") from exc


__all__ = ["SpinBosonBenchmark", "BENCHMARKS", "get_benchmark"]
