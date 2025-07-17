import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def energy(x, coupling_matrix):
    return np.diag(x @ coupling_matrix @ x.T)


def sample_from_ising(coupling_matrix, i0, n_samples=1000):
    n = coupling_matrix.shape[0]
    samples = np.zeros((n_samples, n))
    samples[0] = i0

    for i in range(1, n_samples):
        i_prev = samples[i - 1]
        p_prev = np.exp(energy(np.r_[[i_prev]], coupling_matrix))

        i_proposed = i_prev.copy()
        idx = np.random.randint(0, n)
        i_proposed[idx] = 1 - i_proposed[idx]
        p_proposed = np.exp(energy(np.r_[[i_proposed]], coupling_matrix))

        acceptance_ratio = p_proposed / p_prev
        if np.random.rand() < acceptance_ratio:
            samples[i] = i_proposed
        else:
            samples[i] = i_prev

    return samples

def get_ess(samps):
    samps_hat = samps.mean(axis=0)
    samps_var = samps.var(axis=0)

    def auto_correlation(k):
        rho = ((samps[:-k] - samps_hat) * (samps[k:] - samps_hat)).sum(axis=0) / (
            samps_var * (samps.shape[0] - k)
        )
        return rho

    tau = 1 + 2 * np.array([auto_correlation(k) for k in range(1, 100)]).sum(axis=0)

    ess = samps.shape[0] / tau

    return ess
