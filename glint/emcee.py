import numpy as np
import emcee
import multiprocessing as mp

def make_bounds(param_table: np.ndarray):
    lb = param_table[:, 1].astype(float)
    ub = param_table[:, 2].astype(float)
    x0 = param_table[:, 0].astype(float)
    return x0, lb, ub

def logprior_box(theta, lb, ub):
    if np.any(theta < lb) or np.any(theta > ub):
        return -np.inf
    return 0.0

def run_emcee(logprob_fn, x0, lb, ub, *,
              nwalkers=None, burnin=2000, nstep=100000,
              seed=0, ncpu=32, init_frac=0.2):
    rng = np.random.default_rng(seed)
    ndim = len(x0)
    if nwalkers is None:
        nwalkers = 4 * ndim

    pos = x0 + init_frac * rng.normal(size=(nwalkers, ndim)) * (ub - lb)
    pos = np.clip(pos, lb, ub)

    with mp.Pool(ncpu) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn, pool=pool)
        pos, prob, state = sampler.run_mcmc(pos, burnin, progress=True)
        sampler.reset()
        sampler.run_mcmc(pos, nstep, progress=True)

    return sampler