import jax.numpy as jnp
from jax import lax, random, value_and_grad, vmap
from jax.nn import sigmoid, softmax

from .sampler import run_ising_chain, ising_log_prob, gen_xs


# Surrogate loss
def surrogate_get_conversions(impressions, samples, conversion_prob, population_size):
    marginal_ps = samples.mean(axis=0)
    log_survival = (
        impressions * samples * conversion_prob / (population_size * marginal_ps)
    ).sum(axis=1)
    return jnp.mean(population_size * (1 - jnp.exp(-log_survival)))


def surrogate_objective(
    logit, samples, total_impressions, conversion_prob, population_size
):
    impressions = softmax(logit) * total_impressions
    conversions = surrogate_get_conversions(
        impressions, samples, conversion_prob, population_size
    )
    return -conversions


# Exact loss and gradient
def get_conversions_exact(impressions, M, xs, conversion_prob, population_size):
    lp = vmap(ising_log_prob, in_axes=(0, None))(xs, M)
    ps = softmax(lp)

    marginal_ps = (ps[:, None] * xs).sum(axis=0)
    log_survival = (
        impressions * xs * conversion_prob / (population_size * marginal_ps)
    ).sum(axis=1)
    conversions = population_size * (1 - jnp.exp(-log_survival))
    return jnp.sum(conversions * ps)


def exact_objective(logit, M, xs, total_impressions, conversion_prob, population_size):
    impressions = softmax(logit) * total_impressions
    conversions = get_conversions_exact(
        impressions, M, xs, conversion_prob, population_size
    )
    return -conversions


# Optimizing
def optimize_with_adam(
    M,
    total_impressions,
    conversion_prob,
    population_size,
    method="surrogate",
    options={},
    key=random.PRNGKey(0),
):
    lr = options.get("lr", 0.1)
    beta1 = options.get("beta1", 0.9)
    beta2 = options.get("beta2", 0.999)
    eps = options.get("eps", 1e-8)
    sample_steps = options.get("sample_steps", 10_000)
    adam_steps = options.get("adam_steps", 1_000)

    samples = run_ising_chain(M, None, key, sample_steps)[0]
    logit_init = jnp.zeros_like(conversion_prob)
    n_sites = M.shape[0]

    if method == "exact":
        xs = gen_xs(n_sites)
    elif method == "surrogate":
        samples = run_ising_chain(M, None, key, sample_steps)[0]

    def adam_update(carry, _):
        logit, m, v, t = carry
        if method == "exact":
            loss_val, grad_loss = value_and_grad(
                lambda logit: exact_objective(
                    logit, M, xs, total_impressions, conversion_prob, population_size
                )
            )(logit)
        elif method == "surrogate":
            loss_val, grad_loss = value_and_grad(
                lambda logit: surrogate_objective(
                    logit, samples, total_impressions, conversion_prob, population_size
                )
            )(logit)

        m = beta1 * m + (1 - beta1) * grad_loss
        v = beta2 * v + (1 - beta2) * (grad_loss**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        logit = logit - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return (logit, m, v, t + 1), (logit, loss_val)

    carry_init = (logit_init, jnp.zeros_like(logit_init), jnp.zeros_like(logit_init), 1)

    (carry_final, history) = lax.scan(adam_update, carry_init, None, length=adam_steps)
    return carry_final, history


def optimize_with_sgd(
    M,
    total_impressions,
    conversion_prob,
    population_size,
    method="surrogate",
    options={},
    key=random.PRNGKey(0),
):
    lr = options.get("lr", 0.1)
    sample_steps = options.get("sample_steps", 1_000)
    sgd_steps = options.get("sgd_steps", 1_000)

    logit_init = jnp.zeros_like(conversion_prob)
    n_sites = M.shape[0]

    if method == "exact":
        xs = gen_xs(n_sites)
    elif method == "surrogate":
        samples = run_ising_chain(M, None, key, sample_steps)[0]

    def gradient_decent_update(carry, _):
        logit = carry
        if method == "exact":
            loss_val, grad_loss = value_and_grad(
                lambda logit: exact_objective(
                    logit, M, xs, total_impressions, conversion_prob, population_size
                )
            )(logit)
        elif method == "surrogate":
            loss_val, grad_loss = value_and_grad(
                lambda logit: surrogate_objective(
                    logit, samples, total_impressions, conversion_prob, population_size
                )
            )(logit)

        logit = logit - lr * grad_loss
        return logit, (logit, loss_val)

    (carry_final, history) = lax.scan(
        gradient_decent_update, logit_init, None, length=sgd_steps
    )
    return carry_final, history
