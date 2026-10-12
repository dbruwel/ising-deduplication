"""The Ising model over who watches which channels, and how to sample from it.

A viewer is a 0/1 vector ``x`` with one entry per channel (1 = watches it).
The model puts probability

    p(x) ∝ exp(x^T M x)

on each viewer. Because ``x`` is binary, the diagonal of ``M`` acts as a
per-channel field and the off-diagonal entries as pairwise couplings. This is
the maximum entropy distribution with a given set of pairwise co-viewing
probabilities ``E[x_i x_j]``.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
from jax import Array, lax, random, vmap
from jax.nn import sigmoid, softmax


def ising_log_prob(x: Array, M: Array) -> Array:
    """Unnormalised log probability ``x^T M x`` of a single viewer ``x``."""
    return x.T @ M @ x


def flip_spin(x: Array, i: Array | int) -> Array:
    """Return a copy of ``x`` with entry ``i`` flipped between 0 and 1."""
    return x.at[i].set(1 - x[i])


def mh_update(x: Array, M: Array, key: Array) -> tuple[Array, tuple[Array, Array]]:
    """One Metropolis-Hastings step with a single random spin flip as the proposal.

    Returns
    -------
    x_new : Array
        The state after the step.
    (x, x_prop) : tuple[Array, Array]
        The state before the step and the proposal, used by the surrogate
        gradient in ``fitting.py``.
    """
    n = x.shape[0]
    key_i, key_u = random.split(key)
    i = random.randint(key_i, shape=(), minval=0, maxval=n)

    x_prop = flip_spin(x, i)

    lp = ising_log_prob(x, M)
    lp_prop = ising_log_prob(x_prop, M)
    accept_ratio = jnp.minimum(1.0, jnp.exp(lp_prop - lp))

    u = random.uniform(key_u)
    x_new = jnp.where(u < accept_ratio, x_prop, x)
    return x_new, (x, x_prop)


def run_ising_chain(
    M: Array,
    x0: Array | None = None,
    rng_key: Array | None = None,
    n_steps: int = 1000,
) -> tuple[Array, Array]:
    """Run a Metropolis-Hastings chain on the Ising model.

    Parameters
    ----------
    M : Array
        The ``(n, n)`` interaction matrix.
    x0 : Array, optional
        Starting state. By default each channel is drawn independently with
        probability ``sigmoid(M_ii)``, which is the exact answer when there
        are no couplings.
    rng_key : Array, optional
        JAX random key, defaults to ``PRNGKey(0)``.
    n_steps : int
        Number of steps. There is no burn-in or thinning.

    Returns
    -------
    xs, proposals : tuple[Array, Array]
        Both of shape ``(n_steps, n)``. ``xs[t]`` is the state at step ``t``
        and ``proposals[t]`` the flip proposed from it. ``xs`` on its own is
        the sample.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    key_init, key_chain = random.split(rng_key)
    if x0 is None:
        x0 = random.bernoulli(key_init, sigmoid(jnp.diagonal(M))).astype(int)

    keys = random.split(key_chain, n_steps)

    def scan_fn(x, key):
        return mh_update(x, M, key)

    _, pairs = lax.scan(scan_fn, x0, keys)
    return pairs


def gen_xs(n_sites: int) -> Array:
    """Every possible viewer for ``n_sites`` channels, shape ``(2**n_sites, n_sites)``."""
    return jnp.array(list(itertools.product([0, 1], repeat=n_sites)))


def exact_probs(M: Array, xs: Array | None = None) -> Array:
    """Exact probability of every viewer in ``xs`` (all of them by default)."""
    if xs is None:
        xs = gen_xs(M.shape[0])
    return softmax(vmap(ising_log_prob, in_axes=(0, None))(xs, M))


def exact_moments(M: Array, xs: Array | None = None) -> Array:
    """Exact co-viewing matrix ``E[x_i x_j]`` by enumerating every viewer.

    The diagonal is the probability of watching each channel. Only feasible
    for up to about 20 channels.
    """
    if xs is None:
        xs = gen_xs(M.shape[0])
    ps = exact_probs(M, xs)
    return jnp.einsum("s,si,sj->ij", ps, xs, xs)
