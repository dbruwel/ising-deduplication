import itertools

import jax.numpy as jnp
from jax import lax, random
from jax.nn import sigmoid


def ising_log_prob(x, M):
    return x.T @ M @ x


def flip_spin(x, i):
    return x.at[i].set(1 - x[i])


def mh_update(x, M, key):
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


def run_ising_chain(M, x0=None, rng_key=random.PRNGKey(0), n_steps=1000):
    if x0 is None:
        x0 = random.bernoulli(rng_key, sigmoid(jnp.diagonal(M))).astype(int)

    keys = random.split(rng_key, n_steps)

    def scan_fn(x, key):
        x_new, out = mh_update(x, M, key)
        return x_new, out

    _, pairs = lax.scan(scan_fn, x0, keys)
    return pairs


def gen_xs(n_sites):
    xs = jnp.array(list(itertools.product([0, 1], repeat=n_sites)))
    return xs
