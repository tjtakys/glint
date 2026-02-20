"""
logprob functionは自分専用のnotebookに書く
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
import multiprocessing as mp
import emcee

LogProbFn = Callable[[np.ndarray], float]

def logprior_box(theta, lb, ub):
    """
    ステップ状の事前分布
    """
    if np.any(theta < lb) or np.any(theta > ub):
        return -np.inf
    return 0.0


def make_initial_walkers(x0, lb, ub, nwalkers, seed=0, init_frac=0.2, max_tries=100000):
    """
    x0の周りに walker をばら撒く。範囲外は clip。
    """
    x0 = np.asarray(x0, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)

    rng = np.random.default_rng(seed)
    scale = (ub - lb)

    # pos = x0[None, :] + init_frac * rng.normal(size=(nwalkers, x0.size)) * scale[None, :]
    # return np.clip(pos, lb[None, :], ub[None, :])

    pos = np.empty((nwalkers, x0.size), float)
    n = 0
    tries = 0
    while n < nwalkers:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Failed to sample initial walkers inside bounds. Reduce init_frac or widen bounds.")
        p = x0 + init_frac * rng.normal(size=x0.size) * scale
        if np.all((p >= lb) & (p <= ub)):
            pos[n] = p
            n += 1
    return pos


def run_emcee(logprob_fn, x0, lb, ub, *, nwalkers=None, burnin=2000, production=10000,
             seed=0, ncpu=1, init_frac=0.1, progress=True):
    """
    notebook側で定義した logprob_fn(theta) を使って emcee を回す。
    戻り値：chain, logprob, sampler
    """

    x0 = np.asarray(x0, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)

    ndim = x0.size
    if nwalkers is None:
        nwalkers = 4 * ndim

    pos0 = make_initial_walkers(x0, lb, ub, nwalkers, seed=seed, init_frac=init_frac)

    if ncpu <= 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn)
        state = sampler.run_mcmc(pos0, burnin, progress=progress)
        sampler.reset()
        sampler.run_mcmc(state.coords, production, progress=progress)
    else:
        with mp.Pool(processes=ncpu) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn, pool=pool)
            state = sampler.run_mcmc(pos0, burnin, progress=progress)
            sampler.reset()
            sampler.run_mcmc(state.coords, production, progress=progress)

    chain = sampler.get_chain()
    logprob = sampler.get_log_prob()
    return chain, logprob, sampler