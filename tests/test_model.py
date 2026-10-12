import jax.numpy as jnp
import pytest
from jax import random

from ising_deduplication import (
    exact_moments,
    fit_with_adam,
    gen_xs,
    get_conversions_exact,
    independent_init,
    optimize_with_adam,
    run_ising_chain,
    surrogate_get_conversions,
)
from ising_deduplication.fitting import (
    compute_exact_gradient,
    compute_surrogate_gradient,
)

TARGET = jnp.array([[0.9, 0.78, 0.27], [0.78, 0.8, 0.2], [0.27, 0.2, 0.3]])


def test_independent_init_matches_marginals():
    moments = exact_moments(independent_init(TARGET))
    assert jnp.allclose(moments.diagonal(), TARGET.diagonal(), atol=1e-6)
    p = TARGET.diagonal()
    assert jnp.allclose(moments, jnp.outer(p, p).at[jnp.diag_indices(3)].set(p))


def test_sampler_matches_exact():
    M = jnp.array([[0.5, 1.0, -0.5], [1.0, -1.0, 0.3], [-0.5, 0.3, 0.0]])
    xs = run_ising_chain(M, rng_key=random.PRNGKey(1), n_steps=50_000)[0]
    estimate = (xs[:, :, None] * xs[:, None, :]).mean(axis=0)
    assert jnp.abs(estimate - exact_moments(M)).max() < 0.02


def test_exact_fit_matches_target():
    M, (_, losses) = fit_with_adam(TARGET, method="exact")
    assert losses[-1] < 1e-5
    assert jnp.abs(exact_moments(M) - TARGET).max() < 2e-3


def test_surrogate_gradient_points_the_right_way():
    M = independent_init(TARGET)
    _, exact = compute_exact_gradient(M, TARGET, gen_xs(3))
    # A single chain is noisy, so average a few.
    surrogate = jnp.mean(
        jnp.stack(
            [
                compute_surrogate_gradient(M, TARGET, random.PRNGKey(k), 10_000)[1]
                for k in range(10)
            ]
        ),
        axis=0,
    )
    cos = (exact * surrogate).sum() / (
        jnp.linalg.norm(exact) * jnp.linalg.norm(surrogate)
    )
    assert cos > 0.95


def test_single_channel_reach():
    # One channel, everyone watches it: reach is 1 - exp(-impressions / N).
    M = jnp.array([[20.0]])
    N, T = 1_000.0, 1_500.0
    reach = get_conversions_exact(jnp.array([T]), M, gen_xs(1), jnp.ones(1), N)
    assert reach == pytest.approx(N * (1 - jnp.exp(-T / N)), rel=1e-4)


def test_surrogate_reach_matches_exact():
    M, _ = fit_with_adam(TARGET, method="exact")
    impressions = jnp.array([1e6, 5e5, 2e5])
    q, N = jnp.ones(3) * 0.9, 2e6
    exact = get_conversions_exact(impressions, M, gen_xs(3), q, N)
    samples = run_ising_chain(M, n_steps=50_000)[0]
    assert surrogate_get_conversions(impressions, samples, q, N) == pytest.approx(
        exact, rel=0.01
    )


def test_optimizer_beats_even_split():
    M, _ = fit_with_adam(TARGET, method="exact")
    T, q, N = 2e6, jnp.ones(3) * 0.9, 2e6
    impressions, (_, conversions) = optimize_with_adam(M, T, q, N, method="exact")
    assert impressions.sum() == pytest.approx(T, rel=1e-5)
    even = get_conversions_exact(jnp.ones(3) * T / 3, M, gen_xs(3), q, N)
    best = get_conversions_exact(impressions, M, gen_xs(3), q, N)
    assert best > even
    assert conversions[-1] == pytest.approx(best, rel=1e-4)


def test_unknown_method():
    with pytest.raises(ValueError):
        fit_with_adam(TARGET, method="nope")  # type: ignore[arg-type]
