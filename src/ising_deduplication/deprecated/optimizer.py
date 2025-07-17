import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_to_simplex(v, s=1):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - s
    ind = np.arange(n) + 1
    rho = np.where(u - cssv / ind > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def get_conversions(impressions, i, conversion_prob, population_size):
    log_survival = (impressions * i * conversion_prob / population_size).sum(axis=1)
    return population_size * (1 - np.exp(-log_survival))


def get_grad(impressions, i, conversion_prob, population_size):
    log_survival = (impressions * i * conversion_prob / population_size).sum(axis=1)
    grad = (
        population_size
        * (i * conversion_prob / population_size)
        * np.c_[np.exp(-log_survival)]
    )
    return grad


def optimize(
    total_impressions,
    samps,
    conversion_prob,
    population_size,
    steps=100_000,
    options={},
    method="gradient_decent",
):
    optimiser = optimisers[method]

    channels = samps.shape[1]
    impressions0 = np.ones(channels)

    impressions = project_to_simplex(impressions0, s=total_impressions)

    conversions = get_conversions(
        impressions, samps, conversion_prob, population_size
    ).mean(axis=0)

    hist_impressions, hist_conversions = optimiser(
        impressions,
        conversions,
        total_impressions,
        samps,
        conversion_prob,
        population_size,
        steps,
        options,
    )

    hist_impressions = np.array(hist_impressions)
    hist_conversions = np.array(hist_conversions)

    return hist_impressions, hist_conversions


def _optimize_with_gd(
    impressions,
    conversions,
    total_impressions,
    samps,
    conversion_prob,
    population_size,
    steps,
    options,
):
    theoretical_learning_rate = (
        (total_impressions**2)
        / (population_size * 100_000)
        / (1 - np.exp(-total_impressions * conversion_prob.max() / population_size))
    )

    learning_rate = options.get("learning_rate", theoretical_learning_rate)

    hist_impressions = [impressions]
    hist_conversions = [conversions]

    for _ in range(steps):
        grad = get_grad(impressions, samps, conversion_prob, population_size).mean(
            axis=0
        )

        impressions = project_to_simplex(
            impressions + learning_rate * grad, s=total_impressions
        )
        conversions = get_conversions(
            impressions, samps, conversion_prob, population_size
        ).mean(axis=0)

        hist_impressions.append(impressions)
        hist_conversions.append(conversions)

    return hist_impressions, hist_conversions


def _optimize_with_sa(
    impressions,
    conversions,
    total_impressions,
    samps,
    conversion_prob,
    population_size,
    steps,
    options,
):
    learning_rate = options.get("learning_rate", 10)
    temperature = options.get("temperature", 1)

    channels = samps.shape[1]

    hist_impressions = [impressions]
    hist_conversions = [conversions]

    for _ in range(steps):
        imp_proposed = project_to_simplex(
            impressions + np.random.normal(0, learning_rate, channels),
            s=total_impressions,
        )
        conversions_prop = get_conversions(
            imp_proposed, samps, conversion_prob, population_size
        ).mean(axis=0)

        acceptance_ratio = np.exp(temperature * (conversions_prop - conversions))
        if np.random.rand() < acceptance_ratio:
            impressions = imp_proposed
            conversions = conversions_prop

        hist_impressions.append(impressions)
        hist_conversions.append(conversions)

    return hist_impressions, hist_conversions


def _optimize_with_gsa(
    impressions,
    conversions,
    total_impressions,
    samps,
    conversion_prob,
    population_size,
    steps,
    options,
):
    temperature = options.get("temperature", 1)
    learning_rate = options.get("learning_rate", 10)
    grad_factor = options.get("grad_factor", 0.1)

    channels = samps.shape[1]

    hist_impressions = [impressions]
    hist_conversions = [conversions]

    for _ in range(steps):
        grad = get_grad(impressions, samps, conversion_prob, population_size).mean(
            axis=0
        )

        imp_proposed = project_to_simplex(
            impressions
            + learning_rate * grad_factor * grad
            + np.random.normal(0, temperature, channels),
            s=total_impressions,
        )
        conversions_prop = get_conversions(
            imp_proposed, samps, conversion_prob, population_size
        ).mean(axis=0)

        acceptance_ratio = np.exp(temperature * (conversions_prop - conversions))
        if np.random.rand() < acceptance_ratio:
            impressions = imp_proposed
            conversions = conversions_prop

        hist_impressions.append(impressions)
        hist_conversions.append(conversions)

    return hist_impressions, hist_conversions


optimisers = {
    "gradient_decent": _optimize_with_gd,
    "simulated_anneling": _optimize_with_sa,
    "guided_simulated_anneling": _optimize_with_gsa,
}
