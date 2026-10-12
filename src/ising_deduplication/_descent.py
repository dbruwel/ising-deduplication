"""Plain gradient descent and Adam, shared by ``fitting.py`` and ``optimize.py``."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, lax

# (params, key) -> (loss, gradient of loss with respect to params)
LossAndGrad = Callable[[Array, Array], tuple[Array, Array]]


def adam(
    loss_and_grad: LossAndGrad,
    params: Array,
    keys: Array,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Array, tuple[Array, Array]]:
    """Minimise a loss with Adam, one step per key in ``keys``.

    Returns
    -------
    params : Array
        The final parameters.
    (path, losses) : tuple[Array, Array]
        The parameters after every step, and the loss before every step.
    """

    def step(carry, key):
        params, m, v, t = carry
        loss, grad = loss_and_grad(params, key)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return (params, m, v, t + 1), (params, loss)

    init = (params, jnp.zeros_like(params), jnp.zeros_like(params), 1)
    (params, *_), history = lax.scan(step, init, keys)
    return params, history


def sgd(
    loss_and_grad: LossAndGrad,
    params: Array,
    keys: Array,
    lr: float = 0.1,
) -> tuple[Array, tuple[Array, Array]]:
    """Minimise a loss with plain gradient descent, one step per key in ``keys``.

    Returns the same thing as ``adam``.
    """

    def step(params, key):
        loss, grad = loss_and_grad(params, key)
        params = params - lr * grad
        return params, (params, loss)

    return lax.scan(step, params, keys)
