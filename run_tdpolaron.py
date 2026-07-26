#!/usr/bin/env python3
"""Run constant-bias TD-polaron/static-polaron NZ2 benchmarks.

Examples
--------
Production-resolution model 8 TD-polaron calculation::

    python run_tdpolaron.py --model 8 --method td

Run both transformed frames with a coarser exploratory time step::

    python run_tdpolaron.py --model 8 --method both --dt 0.01 --tmax 5 --plot

Compare against a legacy-format reference trajectory::

    python run_tdpolaron.py --model 8 --method both \
        --reference path/to/reference_rhot.txt --plot

The output text files use the same row-major density-matrix column order as
``run.py``.  The diagonal populations and ``sigma_z`` are invariant under the
polaron transformation.  Off-diagonal entries are polaron-frame coherences.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from Methods.TDPolaronNZ2 import NUMBA_AVAILABLE, propagate_td_polaron_nz2
from Models.SpinBoson_Benchmarks import BENCHMARKS, get_benchmark


@dataclass(frozen=True)
class ReferenceTrajectory:
    times: np.ndarray
    sigma_z: np.ndarray
    label: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the constant-bias spin-boson model with the "
            "time-dependent polaron NZ2 equation."
        )
    )
    parser.add_argument(
        "--model",
        type=int,
        default=8,
        choices=sorted(BENCHMARKS),
        help="legacy spin-boson benchmark number (default: 8)",
    )
    parser.add_argument(
        "--method",
        choices=("td", "static", "both"),
        default="td",
        help="transformed frame(s) to propagate (default: td)",
    )
    parser.add_argument("--dt", type=float, default=None, help="override time step")
    parser.add_argument("--tmax", type=float, default=None, help="override final time")
    parser.add_argument(
        "--nmodes", type=int, default=None, help="override number of bath modes"
    )
    parser.add_argument(
        "--nskip",
        type=int,
        default=None,
        help="output stride in integration steps",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("Results") / "TDPolaron",
        help="output directory",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "numba"),
        default="auto",
        help=(
            "memory-kernel backend; auto uses numba when available "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--renormalize-trace",
        action="store_true",
        help="renormalize trace after every corrector step (off by default)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "optional reference trajectory. Legacy *.txt files must use the "
            "same columns as run.py; CSV files may contain time and sigma_z columns"
        ),
    )
    parser.add_argument(
        "--reference-label",
        default="reference",
        help="legend label for --reference (default: reference)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="write a PNG comparison plot (requires matplotlib)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress reporting"
    )
    return parser.parse_args()


def _progress_printer(label: str):
    last_percent = {"value": -1}

    def callback(done: int, total: int) -> None:
        percent = int(100 * done / total)
        if percent >= last_percent["value"] + 5 or done == total:
            print(f"[{label}] {percent:3d}% ({done}/{total})")
            last_percent["value"] = percent

    return callback


def _load_reference(path: Path, label: str) -> ReferenceTrajectory:
    if not path.is_file():
        raise FileNotFoundError(f"reference trajectory not found: {path}")

    if path.suffix.lower() == ".csv":
        named = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        if named.dtype.names and {"time", "sigma_z"}.issubset(named.dtype.names):
            times = np.atleast_1d(named["time"]).astype(float)
            sigma_z = np.atleast_1d(named["sigma_z"]).astype(float)
        else:
            raise ValueError(
                "reference CSV must have columns named 'time' and 'sigma_z'"
            )
    else:
        data = np.loadtxt(path, dtype=float)
        data = np.atleast_2d(data)
        if data.shape[1] < 9:
            raise ValueError(
                "legacy reference text must contain time plus eight RDM columns"
            )
        times = data[:, 0]
        # Existing repository order: time, Re/Im rho00, rho01, rho10, rho11.
        sigma_z = data[:, 1] - data[:, 7]

    if times.size < 2:
        raise ValueError("reference trajectory must contain at least two points")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(sigma_z)):
        raise ValueError("reference trajectory contains non-finite values")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("reference times must be strictly increasing")

    return ReferenceTrajectory(times=times, sigma_z=sigma_z, label=label)


def _reference_metrics(trajectory, reference: ReferenceTrajectory) -> dict[str, float]:
    start = max(float(trajectory.times[0]), float(reference.times[0]))
    stop = min(float(trajectory.times[-1]), float(reference.times[-1]))
    mask = (reference.times >= start) & (reference.times <= stop)
    if np.count_nonzero(mask) < 2:
        raise ValueError("trajectory and reference have no usable time overlap")

    ref_t = reference.times[mask]
    ref_z = reference.sigma_z[mask]
    calc_z = np.interp(ref_t, trajectory.times, trajectory.sigma_z)
    difference = calc_z - ref_z
    return {
        "reference_points": int(ref_t.size),
        "reference_t_start": float(ref_t[0]),
        "reference_t_stop": float(ref_t[-1]),
        "maximum_absolute_sigma_z_error": float(np.max(np.abs(difference))),
        "rms_sigma_z_error": float(np.sqrt(np.mean(difference**2))),
    }


def _write_metadata(
    path: Path,
    spec,
    bath,
    trajectory,
    backend: str,
    reference_metrics: Optional[dict[str, float]],
) -> None:
    metadata = {
        "model_id": spec.model_id,
        "family": spec.family,
        "legacy_hamiltonian": "H_s = epsilon sigma_z + Delta sigma_x",
        "td_polaron_hamiltonian": "H_s = (bias/2) sigma_z + V sigma_x",
        "epsilon": spec.epsilon,
        "delta": spec.delta,
        "bias": spec.bias,
        "tunneling": spec.tunneling,
        "beta": spec.beta,
        "cutoff": spec.cutoff,
        "strength": spec.strength,
        "n_modes": spec.n_modes,
        "t_max": spec.t_max,
        "dt": spec.dt,
        "nskip": spec.nskip,
        "bath_label": bath.label,
        "log_kappa": bath.log_kappa(),
        "kappa": bath.kappa(),
        "method": trajectory.method,
        "backend": backend,
        "density_matrix_frame": (
            "Schrodinger picture in the chosen polaron frame; diagonal "
            "populations and sigma_z are laboratory-frame invariant"
        ),
        "runtime_seconds": trajectory.runtime_seconds,
        "maximum_trace_error": float(np.max(np.abs(trajectory.trace - 1.0))),
        "minimum_density_eigenvalue": float(np.min(trajectory.minimum_eigenvalue)),
        "final_sigma_z": float(trajectory.sigma_z[-1]),
    }
    if reference_metrics is not None:
        metadata["reference_comparison"] = reference_metrics
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _plot(
    path: Path,
    trajectories: dict[str, object],
    reference: Optional[ReferenceTrajectory],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if reference is not None:
        ax.plot(
            reference.times,
            reference.sigma_z,
            linewidth=2.5,
            label=reference.label,
        )
    for method, trajectory in trajectories.items():
        ax.plot(trajectory.times, trajectory.sigma_z, label=method)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\langle\sigma_z(t)\rangle$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    spec = get_benchmark(args.model).with_overrides(
        t_max=args.tmax,
        dt=args.dt,
        n_modes=args.nmodes,
        nskip=args.nskip,
    )
    if spec.nskip <= 0:
        raise ValueError("nskip must be positive")

    bath = spec.build_bath()
    reference = (
        None
        if args.reference is None
        else _load_reference(args.reference, args.reference_label)
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    methods = ("td", "static") if args.method == "both" else (args.method,)
    trajectories = {}

    print(
        f"Model {spec.model_id}: family={spec.family}, epsilon={spec.epsilon:g}, "
        f"Delta={spec.delta:g}, beta={spec.beta:g}, cutoff={spec.cutoff:g}, "
        f"strength={spec.strength:g}, modes={spec.n_modes}, "
        f"tmax={spec.t_max:g}, dt={spec.dt:g}"
    )
    selected_backend = (
        "numba" if args.backend == "auto" and NUMBA_AVAILABLE else args.backend
    )
    if selected_backend == "auto":
        selected_backend = "numpy"
    print(f"Static Franck-Condon overlap kappa = {bath.kappa():.8e}")
    print(
        f"Backend: {selected_backend}"
        + (
            " (JIT compilation may affect the first run)"
            if selected_backend == "numba"
            else ""
        )
    )
    if reference is not None:
        print(
            f"Reference: {args.reference} "
            f"({reference.times[0]:g} <= t <= {reference.times[-1]:g})"
        )

    wall_start = perf_counter()
    for method in methods:
        callback = None if args.quiet else _progress_printer(method)
        trajectory = propagate_td_polaron_nz2(
            bias=spec.bias,
            tunneling=spec.tunneling,
            bath=bath,
            rho0=spec.rho0,
            t_max=spec.t_max,
            dt=spec.dt,
            method=method,
            hermitize=True,
            renormalize_trace=args.renormalize_trace,
            progress=callback,
            backend=args.backend,
        )
        trajectories[method] = trajectory
        metrics = (
            None if reference is None else _reference_metrics(trajectory, reference)
        )

        stem = f"spinboson_model{spec.model_id}_{method}_polaron"
        trajectory.save_legacy_text(args.outdir / f"{stem}.txt", stride=spec.nskip)
        trajectory.save_csv(args.outdir / f"{stem}.csv", stride=spec.nskip)
        _write_metadata(
            args.outdir / f"{stem}.json",
            spec,
            bath,
            trajectory,
            selected_backend,
            metrics,
        )

        summary = (
            f"[{method}] runtime={trajectory.runtime_seconds:.3f} s, "
            f"max|Tr(rho)-1|={np.max(np.abs(trajectory.trace - 1.0)):.3e}, "
            f"min eig={np.min(trajectory.minimum_eigenvalue):.3e}, "
            f"final <sigma_z>={trajectory.sigma_z[-1]:.8f}"
        )
        if metrics is not None:
            summary += (
                f", max reference error="
                f"{metrics['maximum_absolute_sigma_z_error']:.3e}, "
                f"RMS={metrics['rms_sigma_z_error']:.3e}"
            )
        print(summary)

    if args.plot:
        plot_path = args.outdir / f"spinboson_model{spec.model_id}_polaron_compare.png"
        _plot(plot_path, trajectories, reference)
        print(f"Wrote {plot_path}")

    print(f"Total wall time: {perf_counter() - wall_start:.3f} s")


if __name__ == "__main__":
    main()
