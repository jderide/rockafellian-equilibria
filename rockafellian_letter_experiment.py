"""
Numerical experiment for the Optimization Letters note

    Deride, J. (2026)
    "Diagnosing Infeasibility in Exchange Economies via Rockafellian Relaxation."

This script reproduces the single experiment reported in the note:

  - the multiplicity sanity check for the Shapley-Shubik 2x2 economy
    with r = 2^(8/9) - 2^(1/9), confirming three Walrasian equilibria
    at p* in {1/2, 1, 2};
  - the Rockafellian relaxation on the stressed variant
    (e1 = (0.50, 0.05), e2 = (0.05, 0.50)), sweeping the penalty
    parameter lambda in [10^0, 10^7] and recording the residual
    u*(lambda) = Z(x*(lambda))_+ together with its l2-norm;
  - the figure fig_stressed_letter.{pdf,png} cited in Section 3.

The penalty is the inequality form (lambda/2) ||Z(x)_+||^2 used in
Theorem 3 of the note. For the symmetric stressed economy considered
here Z(x) >= 0 holds at every iterate, so ||Z(x)_+||^2 = ||Z(x)||^2
numerically.

Requirements: Python 3.10+, numpy, scipy, matplotlib.

Usage (from the directory containing this file):
    python rockafellian_letter_experiment.py
which writes:
    fig_stressed_letter.pdf
    fig_stressed_letter.png
in the current directory, alongside a brief printed summary.

A custom output directory can be supplied as the first command-line
argument:
    python rockafellian_letter_experiment.py /path/to/output/dir
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# Plot aesthetics
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
NAVY, RUST, MOSS, GOLD, SLATE = "#1f4e79", "#b43a3a", "#3a7d44", "#af7b2a", "#555555"


# ----------------------------------------------------------------------
# Economy parameters
# ----------------------------------------------------------------------
SURVIVAL = 0.3   # x_il >= 0.3 (survival floor)
SATIATION = 3.0  # x_il <= 3.0 (satiation ceiling)
EXPONENT = 8     # quasi-linear utility exponent

# Baseline endowments produce three Walrasian equilibria at p* in {1/2, 1, 2}
R_BASE = 2 ** (8 / 9) - 2 ** (1 / 9)

# Stressed endowments: aggregate (0.55, 0.55) < 2 * survival = (0.6, 0.6).
# The feasibility set {x in [0.3, 3]^4 : Z(x) <= 0} is empty.
E1_STRESSED = np.array([0.50, 0.05])
E2_STRESSED = np.array([0.05, 0.50])

# Penalty schedule for the Rockafellian relaxation
LAMBDAS = np.logspace(0, 7, 36)


# ----------------------------------------------------------------------
# Economy primitives
# ----------------------------------------------------------------------
def utility(x: np.ndarray) -> float:
    """Aggregate utility U(x) = u_1(x_1) + u_2(x_2) on the survival set."""
    x11, x21, x12, x22 = x
    if not all(SURVIVAL <= xi <= SATIATION for xi in (x11, x21, x12, x22)):
        return -np.inf
    return (x11 - x21 ** (-EXPONENT) / EXPONENT
            + x22 - x12 ** (-EXPONENT) / EXPONENT)


def excess_demand(x: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Aggregate excess demand Z(x) = sum_i (x_i - e^i)."""
    x11, x21, x12, x22 = x
    return np.array([(x11 + x12) - (e1[0] + e2[0]),
                     (x21 + x22) - (e1[1] + e2[1])])


def excess_demand_price(p: float, r: float) -> float:
    """
    Closed-form good-2 excess demand for the baseline economy as a
    function of the price ratio p = p1/p2 (Walras' law makes good 1
    redundant). Used only by verify_three_equilibria.
    """
    return p ** (1 / 9) + p * r + 2.0 - p ** (8 / 9) - r - 2.0


# ----------------------------------------------------------------------
# Sanity check: three Walrasian equilibria in the baseline economy
# ----------------------------------------------------------------------
def verify_three_equilibria() -> None:
    """Confirm that p* in {1/2, 1, 2} are roots of Z_2(p) when r = R_BASE."""
    for p_eq in (0.5, 1.0, 2.0):
        val = excess_demand_price(p_eq, R_BASE)
        assert abs(val) < 1e-12, f"Z(p={p_eq}) = {val}, expected 0"
    print(f"[sanity] Three baseline equilibria verified to machine precision "
          f"(r = 2^(8/9) - 2^(1/9) = {R_BASE:.6f}).")


# ----------------------------------------------------------------------
# Rockafellian primal relaxation on the stressed economy
# ----------------------------------------------------------------------
def solve_rockafellian_sweep(
    e1: np.ndarray,
    e2: np.ndarray,
    lambdas: np.ndarray,
    x0: np.ndarray | None = None,
) -> dict:
    """
    Solve, for each lambda in `lambdas`, the relaxed problem

        min_{x in X}  -U(x) + (lambda/2) ||Z(x)_+||^2_2,

    where (.)_+ is componentwise positive part. Returns a dict
    of arrays containing the lambda schedule, the optimal allocations
    x*(lambda), the residuals u*(lambda) = Z(x*(lambda))_+ and their
    l2-norms.

    Iterates are warm-started from the previous lambda, with the
    initial guess at a uniform interior allocation.
    """
    if x0 is None:
        x0 = np.full(4, 0.4)
    bounds = [(SURVIVAL, SATIATION)] * 4

    xs, us, devs = [], [], []
    x_warm = x0.copy()

    for lam in lambdas:
        def objective(x_vec, lam=lam):
            U = utility(x_vec)
            if not np.isfinite(U):
                return 1e12
            z = excess_demand(x_vec, e1, e2)
            z_plus = np.maximum(z, 0.0)
            return -U + 0.5 * lam * float(z_plus @ z_plus)

        res = minimize(
            objective, x_warm, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 800},
        )
        x_star = res.x
        z_plus = np.maximum(excess_demand(x_star, e1, e2), 0.0)
        xs.append(x_star)
        us.append(z_plus)
        devs.append(float(np.linalg.norm(z_plus)))
        x_warm = x_star

    return {
        "lambdas": np.asarray(lambdas, dtype=float),
        "xs": np.asarray(xs),
        "us": np.asarray(us),
        "deviations": np.asarray(devs),
    }


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_fig_stressed_letter(trace: dict, out_base: str) -> None:
    """Two-panel figure for the OL note (no figure-level suptitle)."""
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })
    MOSS, GOLD, RUST, SLATE = "#3a7d44", "#af7b2a", "#b43a3a", "#555555"

    lambdas = trace["lambdas"]
    us = trace["us"]
    devs = trace["deviations"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

    axes[0].semilogx(lambdas, us[:, 0], "s-", color=MOSS, lw=1.3, ms=4,
                     label=r"$v^*_1(\lambda)$ (good 1 shortage)")
    axes[0].semilogx(lambdas, us[:, 1], "^-", color=GOLD, lw=1.3, ms=4,
                     label=r"$v^*_2(\lambda)$ (good 2 shortage)")
    axes[0].axhline(0.0, color=SLATE, lw=0.7)
    axes[0].axhline(0.05, color=RUST, ls="--", lw=1.2,
                    label=fr"analytical floor $v=0.05$")
    axes[0].set_xlabel(r"Rockafellian penalty $\lambda$")
    axes[0].set_ylabel(r"supply perturbation $v^*(\lambda)$")
    axes[0].set_title("Components of the residual")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(lambdas, devs, "s-", color=MOSS, lw=1.4, ms=4.5)
    axes[1].axhline(devs[-1], color=RUST, ls="--", lw=1.2,
                    label=fr"$\|v^*_\infty\|_2 \approx {devs[-1]:.3f}$")
    axes[1].set_xlabel(r"Rockafellian penalty $\lambda$")
    axes[1].set_ylabel(r"$\|v^*(\lambda)\|_2$")
    axes[1].set_title("Convergence to the infeasibility floor")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    plt.close(fig)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    verify_three_equilibria()

    print(f"[run]    Rockafellian sweep on stressed economy, "
          f"lambda in [{LAMBDAS[0]:.0e}, {LAMBDAS[-1]:.0e}] "
          f"({len(LAMBDAS)} points).")
    trace = solve_rockafellian_sweep(E1_STRESSED, E2_STRESSED, LAMBDAS)

    # Analytical infeasibility floor: per-good shortfall is
    #   (sum of survivals) - (sum of endowments) = 0.6 - 0.55 = 0.05,
    # so ||u*_inf||_2 = sqrt(2) * 0.05 ~ 0.0707.
    analytic_floor = float(np.sqrt(2) * 0.05)
    final_norm = float(trace["deviations"][-1])

    print(f"[result] ||v*_inf||_2 (numerical) = {final_norm:.6f}")
    print(f"[result] sqrt(2) * 0.05           = {analytic_floor:.6f}")
    print(f"[result] absolute gap             = "
          f"{abs(final_norm - analytic_floor):.2e}")

    out_base = os.path.join(out_dir, "fig_stressed_letter")
    plot_fig_stressed_letter(trace, out_base)
    print(f"[write]  {out_base}.pdf")
    print(f"[write]  {out_base}.png")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.abspath(__file__))
    main(out_dir)
