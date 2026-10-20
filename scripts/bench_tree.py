"""Time mob() with LmModel and GlmModel (binomial) on simulated data.

Usage, from the repository root:

    uv run python scripts/bench_tree.py [--sizes 1000 5000 20000] [--runs 3]
        [--profile-top 12]

The data has 6 partitioning variables: z1 to z4 numeric, z5 ordered (5 levels)
and z6 nominal (5 levels). The coefficients change at z1 <= 0, at z5 (ordered)
on the left and at z6 (nominal) on the right, so the tree grows several levels.
Nothing is optimized or tuned: this measures the library as it is.
"""

import argparse
import cProfile
import io
import pstats
import statistics
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from modeltrees.instability import VariableKind
from modeltrees.models.glm import GlmModel
from modeltrees.models.lm import LmModel
from modeltrees.tree import ModelTree, Variable, mob

SEED = 20261101
ORDERED_LEVELS = tuple(f"o{i}" for i in range(1, 6))
NOMINAL_LEVELS = tuple("abcde")


@dataclass
class Data:
    n: int
    x: np.ndarray
    z: dict[str, Variable]
    y_gauss: np.ndarray
    y_bin: np.ndarray


def simulate(n: int, seed: int = SEED) -> Data:
    rng = np.random.default_rng(seed + n)
    xv = rng.standard_normal(n)
    z1, z2, z3, z4 = (rng.uniform(-1, 1, n) for _ in range(4))
    z5 = rng.integers(0, 5, n)
    z6 = rng.integers(0, 5, n)
    low5 = z5 <= 1
    in_acd = np.isin(z6, [0, 2, 3])
    # z1 <= 0: slope depends on the ordered z5. z1 > 0: it depends on nominal z6.
    b0 = np.where(z1 <= 0, np.where(low5, 1.0, -1.5), np.where(in_acd, 2.0, 0.0))
    b1 = np.where(z1 <= 0, np.where(low5, 1.0, 0.0), np.where(in_acd, -1.0, 1.5))
    eta = b0 + b1 * xv
    y_gauss = eta + rng.standard_normal(n)
    y_bin = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(np.float64)
    z = {
        "z1": Variable("z1", VariableKind.NUMERIC, z1),
        "z2": Variable("z2", VariableKind.NUMERIC, z2),
        "z3": Variable("z3", VariableKind.NUMERIC, z3),
        "z4": Variable("z4", VariableKind.NUMERIC, z4),
        "z5": Variable("z5", VariableKind.ORDERED, z5, ORDERED_LEVELS),
        "z6": Variable("z6", VariableKind.NOMINAL, z6, NOMINAL_LEVELS),
    }
    return Data(n, np.column_stack([np.ones(n), xv]), z, y_gauss, y_bin)


def runner(name: str, d: Data) -> Callable[[], ModelTree]:
    def run() -> ModelTree:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "lm":
                return mob(LmModel(), d.y_gauss, d.x, d.z)
            return mob(GlmModel("binomial"), d.y_bin, d.x, d.z)

    return run


def time_runs(run: Callable[[], ModelTree], runs: int) -> tuple[list[float], ModelTree]:
    times = []
    tree = run()  # warm-up, not timed
    for _ in range(runs):
        t0 = time.perf_counter()
        tree = run()
        times.append(time.perf_counter() - t0)
    return times, tree


def profile(run: Callable[[], ModelTree], top: int) -> str:
    prof = cProfile.Profile()
    prof.enable()
    run()
    prof.disable()
    out = io.StringIO()
    stats = pstats.Stats(prof, stream=out).strip_dirs()
    total = stats.total_tt  # type: ignore[attr-defined]
    out.write(f"profiled run: {total:.2f} s under cProfile\n\n")
    out.write("by cumulative time\n")
    stats.sort_stats("cumulative").print_stats(top)
    out.write("by own time\n")
    stats.sort_stats("tottime").print_stats(top)
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--profile-top", type=int, default=12)
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    print(f"numpy {np.__version__}, runs per cell {args.runs} (after one warm-up)\n")
    print("model  n       nodes leaves depth  min s   median s  max s")
    profiles = []
    for n in args.sizes:
        d = simulate(n)
        for name in ("lm", "glm"):
            run = runner(name, d)
            times, tree = time_runs(run, args.runs)
            leaves = sum(nd.terminal for nd in tree.nodes)
            depth = max(nd.depth for nd in tree.nodes)
            print(
                f"{name:<6} {n:<7} {len(tree.nodes):<5} {leaves:<6} {depth:<5} "
                f"{min(times):<7.3f} {statistics.median(times):<9.3f} {max(times):.3f}"
            )
            if not args.no_profile:
                profiles.append((name, n, profile(run, args.profile_top)))
    for name, n, text in profiles:
        print(f"\n=== profile: {name}, n = {n} ===\n{text}")


if __name__ == "__main__":
    main()
