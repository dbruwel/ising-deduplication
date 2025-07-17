import jax.numpy as jnp
from jax import lax, random, value_and_grad, vmap
from jax.nn import softmax

from .sampler import gen_xs, ising_log_prob, run_ising_chain


# Surrogate loss and gradient
def objective(x):
    return jnp.outer(x, x)


def alpha(x, x_prime, M):
    lp_x = ising_log_prob(x, M)
    lp_xp = ising_log_prob(x_prime, M)
    return jnp.minimum(1.0, jnp.exp(lp_xp - lp_x))


def surrogate_loss(pairs, M, target):
    def contribution(x_xp):
        x, x_prime = x_xp
        a = alpha(x, x_prime, M)
        return a * objective(x_prime) + (1 - a) * objective(x)

    return ((jnp.mean(vmap(contribution)(pairs), axis=0) - target) ** 2).sum()


def compute_surrogate_gradient(M, target, key, n_steps):
    pairs = run_ising_chain(M, None, key, n_steps)

    loss = lambda M: surrogate_loss(pairs, M, target)
    loss_val, grad_loss = value_and_grad(loss)(M)
    return loss_val, grad_loss


# Exact loss and gradient
def exact_loss(xs, M, target):
    lp = vmap(ising_log_prob, in_axes=(0, None))(xs, M)
    ps = softmax(lp)

    true_prob = (ps[:, None, None] * xs[:, :, None] * xs[:, None, :]).sum(axis=0)

    return ((true_prob - target) ** 2).sum()


def compute_exact_gradient(M, target, xs):
    loss = lambda M: exact_loss(xs, M, target)
    loss_val, grad_loss = value_and_grad(loss)(M)
    return loss_val, grad_loss


# Fitting
def fit_with_adam(target, method="surrogate", options={}, key=random.PRNGKey(0)):
    lr = options.get("lr", 0.1)
    beta1 = options.get("beta1", 0.9)
    beta2 = options.get("beta2", 0.999)
    eps = options.get("eps", 1e-8)
    sample_steps = options.get("sample_steps", 10_000)
    adam_steps = options.get("adam_steps", 1_000)

    n_sites = target.shape[0]
    M_init = jnp.diag(jnp.log(target.diagonal() / (1 - target.diagonal())))
    keys = random.split(key, adam_steps)

    if method == "exact":
        xs = gen_xs(n_sites)

    def adam_update(carry, key):
        M, m, v, t = carry
        if method == "exact":
            loss_val, grad_loss = compute_exact_gradient(M, target, xs)
        elif method == "surrogate":
            loss_val, grad_loss = compute_surrogate_gradient(
                M, target, key, sample_steps
            )
        m = beta1 * m + (1 - beta1) * grad_loss
        v = beta2 * v + (1 - beta2) * (grad_loss**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        M = M - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return (M, m, v, t + 1), (M, loss_val)

    carry_init = (M_init, jnp.zeros_like(M_init), jnp.zeros_like(M_init), 1)

    (carry_final, history) = lax.scan(adam_update, carry_init, keys)
    return carry_final, history


def fit_with_sgd(target, method="surrogate", options={}, key=random.PRNGKey(0)):
    lr = options.get("lr", 0.1)
    sample_steps = options.get("sample_steps", 1_000)
    sgd_steps = options.get("sgd_steps", 1_000)

    n_sites = target.shape[0]
    M_init = jnp.diag(jnp.log(target.diagonal() / (1 - target.diagonal())))
    keys = random.split(key, sgd_steps)

    if method == "exact":
        xs = gen_xs(n_sites)

    def gradient_decent_update(carry, key):
        M = carry
        if method == "exact":
            loss_val, grad_loss = compute_exact_gradient(M, target, xs)
        elif method == "surrogate":
            loss_val, grad_loss = compute_surrogate_gradient(
                M, target, key, sample_steps
            )
        M = M - lr * grad_loss
        return M, (M, loss_val)

    (carry_final, history) = lax.scan(gradient_decent_update, M_init, keys)
    return carry_final, history
