"""Fitting the Ising interaction matrix to a co-viewing matrix.

The target is an ``(n, n)`` matrix of probabilities: ``target[i, j]`` is the
share of people who watch both channel ``i`` and channel ``j``, and the
diagonal is the share who watch each channel. We look for the ``M`` whose
model co-viewing matrix ``E[x_i x_j]`` matches it, by minimising the sum of
squared differences.

There are two ways of getting the loss and its gradient:

- ``"exact"`` enumerates all ``2**n`` viewers. Exact and fast for a handful of
  channels, unusable past about 20.
- ``"surrogate"`` estimates the moments from a Metropolis-Hastings chain and
  differentiates through the acceptance probability (see ``surrogate_loss``).
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array, random, value_and_grad, vmap

from ._descent import adam, sgd
from .sampler import exact_moments, gen_xs, ising_log_prob, run_ising_chain

Method = Literal["surrogate", "exact"]


# Surrogate loss and gradient
def objective(x: Array) -> Array:
    """The quantity being matched for one viewer, ``x x^T``."""
    return jnp.outer(x, x)


def alpha(x: Array, x_prime: Array, M: Array) -> Array:
    """Metropolis-Hastings acceptance probability for moving from ``x`` to ``x_prime``."""
    lp_x = ising_log_prob(x, M)
    lp_xp = ising_log_prob(x_prime, M)
    return jnp.minimum(1.0, jnp.exp(lp_xp - lp_x))


def surrogate_loss(pairs: tuple[Array, Array], M: Array, target: Array) -> Array:
    """Squared error of the co-viewing matrix, estimated from an MCMC chain.

    For each step of the chain, from state ``x`` with proposal ``x'``, the next
    state is ``x'`` with probability ``alpha`` and ``x`` otherwise, so

        alpha * x' x'^T + (1 - alpha) * x x^T

    is the expected value of the next sample's ``x x^T``. Averaging this over
    the chain gives a (Rao-Blackwellised) estimate of ``E[x x^T]`` that depends
    smoothly on ``M`` through ``alpha``, so it can be differentiated.

    The states ``x`` themselves are held fixed, so the gradient ignores how the
    stationary distribution moves with ``M``. It isn't the exact gradient, but
    it points in close to the same direction (see ``notes/findings.md``).
    """

    def contribution(x_xp):
        x, x_prime = x_xp
        a = alpha(x, x_prime, M)
        return a * objective(x_prime) + (1 - a) * objective(x)

    return ((jnp.mean(vmap(contribution)(pairs), axis=0) - target) ** 2).sum()


def compute_surrogate_gradient(
    M: Array, target: Array, key: Array, n_steps: int
) -> tuple[Array, Array]:
    """Run a fresh chain of ``n_steps`` and return the surrogate loss and its gradient."""
    pairs = run_ising_chain(M, None, key, n_steps)
    return value_and_grad(lambda M: surrogate_loss(pairs, M, target))(M)


# Exact loss and gradient
def exact_loss(xs: Array, M: Array, target: Array) -> Array:
    """Squared error of the co-viewing matrix, summing over every viewer in ``xs``."""
    return ((exact_moments(M, xs) - target) ** 2).sum()


def compute_exact_gradient(M: Array, target: Array, xs: Array) -> tuple[Array, Array]:
    """Return the exact loss and its gradient."""
    return value_and_grad(lambda M: exact_loss(xs, M, target))(M)


# Fitting
def independent_init(target: Array) -> Array:
    """The ``M`` that matches the diagonal of ``target`` with no couplings.

    With no couplings each channel is independent with probability
    ``sigmoid(M_ii)``, so ``M_ii = logit(target_ii)``.
    """
    p = target.diagonal()
    return jnp.diag(jnp.log(p / (1 - p)))


def _loss_and_grad(target: Array, method: Method, sample_steps: int):
    if method == "exact":
        xs = gen_xs(target.shape[0])
        return lambda M, _key: compute_exact_gradient(M, target, xs)
    if method == "surrogate":
        return lambda M, key: compute_surrogate_gradient(M, target, key, sample_steps)
    raise ValueError(f"method should be 'exact' or 'surrogate', not {method!r}")


def fit_with_adam(
    target: Array,
    method: Method = "surrogate",
    *,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    sample_steps: int = 10_000,
    adam_steps: int = 1_000,
    key: Array | None = None,
) -> tuple[Array, tuple[Array, Array]]:
    """Fit ``M`` to a co-viewing matrix with Adam.

    Starts from ``independent_init(target)``.

    Parameters
    ----------
    target : Array
        ``(n, n)`` symmetric matrix of co-viewing probabilities, with the
        share watching each channel on the diagonal.
    method : {"surrogate", "exact"}
        How to compute the loss and gradient (see the module docstring).
    lr, beta1, beta2, eps : float
        Adam settings. ``lr=0.1`` is fine for a handful of channels but too
        big for more (at 14 channels it diverges); 0.01 with more steps is
        safer there.
    sample_steps : int
        Length of the MCMC chain used for each step (surrogate only).
    adam_steps : int
        Number of Adam steps.
    key : Array, optional
        JAX random key, defaults to ``PRNGKey(0)``.

    Returns
    -------
    M : Array
        The final interaction matrix. With the surrogate method this is noisy;
        averaging the last few hundred entries of ``Ms`` is usually closer.
    (Ms, losses) : tuple[Array, Array]
        ``M`` after every step, and the loss before every step.
    """
    if key is None:
        key = random.PRNGKey(0)
    keys = random.split(key, adam_steps)
    loss_and_grad = _loss_and_grad(target, method, sample_steps)
    return adam(
        loss_and_grad,
        independent_init(target),
        keys,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )


def fit_with_sgd(
    target: Array,
    method: Method = "surrogate",
    *,
    lr: float = 0.1,
    sample_steps: int = 1_000,
    sgd_steps: int = 1_000,
    key: Array | None = None,
) -> tuple[Array, tuple[Array, Array]]:
    """Fit ``M`` to a co-viewing matrix with plain gradient descent.

    Same as ``fit_with_adam`` otherwise. The loss is small and flat, so this
    needs a lot more steps (or a bigger ``lr``) than Adam does.
    """
    if key is None:
        key = random.PRNGKey(0)
    keys = random.split(key, sgd_steps)
    loss_and_grad = _loss_and_grad(target, method, sample_steps)
    return sgd(loss_and_grad, independent_init(target), keys, lr=lr)
