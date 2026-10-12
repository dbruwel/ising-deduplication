"""Deduplicating reach across channels with a maximum entropy (Ising) model.

A small side project. The model and sampler live in ``sampler.py``, fitting it
to a co-viewing matrix in ``fitting.py``, and splitting a budget across
channels in ``optimize.py``.
"""

from .fitting import fit_with_adam, fit_with_sgd, independent_init
from .optimize import (
    get_conversions_exact,
    optimize_with_adam,
    optimize_with_sgd,
    surrogate_get_conversions,
)
from .sampler import exact_moments, exact_probs, gen_xs, run_ising_chain

__all__ = [
    "exact_moments",
    "exact_probs",
    "fit_with_adam",
    "fit_with_sgd",
    "gen_xs",
    "get_conversions_exact",
    "independent_init",
    "optimize_with_adam",
    "optimize_with_sgd",
    "run_ising_chain",
    "surrogate_get_conversions",
]
