"""Splitting a budget of impressions across channels to maximise reach.

The model: a population of ``population_size`` people, with each person's
channels ``x`` drawn from the Ising model. Impressions bought on channel ``j``
are spread evenly (Poisson) over the ``population_size * p_j`` people who
watch it, where ``p_j`` is the share watching channel ``j``. Each impression
converts with probability ``conversion_prob[j]``. A person with channels ``x``
is then converted at least once with probability

    1 - exp(-sum_j x_j * impressions_j * conversion_prob_j / (population_size * p_j))

and the expected number of people converted is ``population_size`` times the
average of that over ``x``. With ``conversion_prob = 1`` this is plain reach
(people seeing at least one impression).

The allocation is parameterised as ``softmax(logit) * total_impressions`` so it
always adds up to the budget, and the logits are optimised with SGD or Adam.
As with fitting, the average over ``x`` is either over every viewer
(``"exact"``) or over a fixed MCMC sample (``"surrogate"``).
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array, random, value_and_grad
from jax.nn import softmax

from ._descent import adam, sgd
from .sampler import exact_probs, gen_xs, run_ising_chain

Method = Literal["surrogate", "exact"]


def _p_converted(
    impressions: Array,
    xs: Array,
    conversion_prob: Array,
    population_size: float,
    marginal_ps: Array,
) -> Array:
    """Probability each viewer in ``xs`` is converted at least once."""
    rate = (impressions * xs * conversion_prob / (population_size * marginal_ps)).sum(
        axis=1
    )
    return 1 - jnp.exp(-rate)


# Surrogate loss
def surrogate_get_conversions(
    impressions: Array,
    samples: Array,
    conversion_prob: Array,
    population_size: float,
) -> Array:
    """Expected number of people converted, averaging over a sample of viewers.

    The share watching each channel is also estimated from ``samples``.
    """
    marginal_ps = samples.mean(axis=0)
    p = _p_converted(
        impressions, samples, conversion_prob, population_size, marginal_ps
    )
    return population_size * jnp.mean(p)


def surrogate_objective(
    logit: Array,
    samples: Array,
    total_impressions: float,
    conversion_prob: Array,
    population_size: float,
) -> Array:
    """Minus the share of people converted by the split ``softmax(logit)``, from a sample."""
    impressions = softmax(logit) * total_impressions
    conversions = surrogate_get_conversions(
        impressions, samples, conversion_prob, population_size
    )
    return -conversions / population_size


# Exact loss
def get_conversions_exact(
    impressions: Array,
    M: Array,
    xs: Array,
    conversion_prob: Array,
    population_size: float,
) -> Array:
    """Expected number of people converted, summing over every viewer in ``xs``."""
    ps = exact_probs(M, xs)
    marginal_ps = (ps[:, None] * xs).sum(axis=0)
    p = _p_converted(impressions, xs, conversion_prob, population_size, marginal_ps)
    return population_size * jnp.sum(p * ps)


def exact_objective(
    logit: Array,
    M: Array,
    xs: Array,
    total_impressions: float,
    conversion_prob: Array,
    population_size: float,
) -> Array:
    """Minus the share of people converted by the split ``softmax(logit)``, exactly."""
    impressions = softmax(logit) * total_impressions
    conversions = get_conversions_exact(
        impressions, M, xs, conversion_prob, population_size
    )
    return -conversions / population_size


# Optimizing
def _loss_and_grad(
    M: Array,
    total_impressions: float,
    conversion_prob: Array,
    population_size: float,
    method: Method,
    sample_steps: int,
    key: Array,
):
    if method == "exact":
        xs = gen_xs(M.shape[0])

        def loss(logit):
            return exact_objective(
                logit, M, xs, total_impressions, conversion_prob, population_size
            )

    elif method == "surrogate":
        samples = run_ising_chain(M, None, key, sample_steps)[0]

        def loss(logit):
            return surrogate_objective(
                logit, samples, total_impressions, conversion_prob, population_size
            )

    else:
        raise ValueError(f"method should be 'exact' or 'surrogate', not {method!r}")

    loss_and_grad = value_and_grad(loss)
    return lambda logit, _: loss_and_grad(logit)


def _to_impressions(
    logits: Array, losses: Array, total_impressions: float, population_size: float
) -> tuple[Array, tuple[Array, Array]]:
    path = softmax(logits, axis=-1) * total_impressions
    return path[-1], (path, -losses * population_size)


def optimize_with_adam(
    M: Array,
    total_impressions: float,
    conversion_prob: Array,
    population_size: float,
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
    """Find the split of ``total_impressions`` across channels that converts the most people.

    Starts from an even split.

    Parameters
    ----------
    M : Array
        The fitted ``(n, n)`` interaction matrix.
    total_impressions : float
        The budget, in impressions.
    conversion_prob : Array
        Probability that a single impression on each channel converts.
    population_size : float
        Number of people in the population.
    method : {"surrogate", "exact"}
        Average over a fixed MCMC sample of ``sample_steps`` viewers, or over
        every possible viewer.
    lr, beta1, beta2, eps : float
        Adam settings, applied to the logits of the split.
    sample_steps : int
        Size of the MCMC sample (surrogate only).
    adam_steps : int
        Number of Adam steps.
    key : Array, optional
        JAX random key for the MCMC sample, defaults to ``PRNGKey(0)``.

    Returns
    -------
    impressions : Array
        The final impressions on each channel.
    (impressions_path, conversions) : tuple[Array, Array]
        The split after every step, and the expected conversions before every
        step.
    """
    if key is None:
        key = random.PRNGKey(0)
    loss_and_grad = _loss_and_grad(
        M,
        total_impressions,
        conversion_prob,
        population_size,
        method,
        sample_steps,
        key,
    )
    logit = jnp.zeros(M.shape[0])
    _, (logits, losses) = adam(
        loss_and_grad,
        logit,
        jnp.arange(adam_steps),
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    return _to_impressions(logits, losses, total_impressions, population_size)


def optimize_with_sgd(
    M: Array,
    total_impressions: float,
    conversion_prob: Array,
    population_size: float,
    method: Method = "surrogate",
    *,
    lr: float = 10.0,
    sample_steps: int = 1_000,
    sgd_steps: int = 1_000,
    key: Array | None = None,
) -> tuple[Array, tuple[Array, Array]]:
    """Same as ``optimize_with_adam`` but with plain gradient descent.

    The loss is the share of the population converted, so the gradient is
    small. When the budget is small compared to the population, reach is close
    to linear in impressions and the loss is very flat, and this can take a lot
    more than 1000 steps to converge. Adam doesn't have that problem.
    """
    if key is None:
        key = random.PRNGKey(0)
    loss_and_grad = _loss_and_grad(
        M,
        total_impressions,
        conversion_prob,
        population_size,
        method,
        sample_steps,
        key,
    )
    logit = jnp.zeros(M.shape[0])
    _, (logits, losses) = sgd(loss_and_grad, logit, jnp.arange(sgd_steps), lr=lr)
    return _to_impressions(logits, losses, total_impressions, population_size)
