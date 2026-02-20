"""
logprob functionは個々のnotebookに書く
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
    x0の周りに walker をばら撒く。範囲内に収まるまで繰り返す（clipだと端に固定化されることがあるので）
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
             seed=0, ncpu=1, init_frac=0.1, skip_initial_state_check=False, progress=True):
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
    # --- burn-in前のパラメータ分散をチェック ---
    width = ub - lb
    spread0 = np.std(pos0, axis=0) / width  # walker間の相対的なばらつき（各パラメータごと）
    print(f"[init] min std over dims = {spread0.min():.3e}, median std = {np.median(spread0):.3e}")

    if ncpu <= 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn)
        state = sampler.run_mcmc(pos0, burnin, progress=progress)
        sampler.reset()
        sampler.run_mcmc(state.coords, production, progress=progress)
    else:
        with mp.Pool(processes=ncpu) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn, pool=pool)
            
            # --- burn-in ---
            print(f"Running burn-in with {ncpu} CPUs...")
            state = sampler.run_mcmc(pos0, burnin, progress=progress)
            
            # --- burn-in後のパラメータ分散をチェック ---
            acc = np.mean(sampler.acceptance_fraction)

            spread = np.std(state.coords, axis=0) / width  # walker間の相対的なばらつき（各パラメータごと）
            print(f"[burnin] acceptance mean = {acc:.3f}")
            print(f"[burnin] min std over dims = {spread.min():.3e}, median std = {np.median(spread):.3e}")
            sampler.reset()

            # --- production ---
            print(f"Running production with {ncpu} CPUs...")
            # burn-inですでに結構収束していると、initial state checkで線型独立になってない、というエラーになることがある。これをスキップするオプションを追加。
            sampler.run_mcmc(state.coords, production, skip_initial_state_check=skip_initial_state_check, progress=progress)

    
    # ==========================
    # Post-run summary (production)
    # ==========================
    chain = sampler.get_chain()          # (nsteps, nwalkers, ndim)
    logprob = sampler.get_log_prob()     # (nsteps, nwalkers)

    nsteps, nwalkers, ndim = chain.shape
    acc_all = sampler.acceptance_fraction
    print("\n===== MCMC summary =====")
    print(f"steps (production): {nsteps} | walkers: {nwalkers} | ndim: {ndim}")
    print(f"acceptance: mean={np.mean(acc_all):.3f}, median={np.median(acc_all):.3f}, "
          f"min={np.min(acc_all):.3f}, max={np.max(acc_all):.3f}")

    # last-step logprob stats across walkers
    lp_last = logprob[-1]
    finite_last = np.isfinite(lp_last)
    print(f"logprob (last step): finite fraction={np.mean(finite_last):.3f}, "
          f"median={np.median(lp_last[finite_last]):.3f}, max={np.max(lp_last[finite_last]):.3f}")

    # show worst walkers by last-step logprob (useful for diagnosing stuck/outliers)
    k = min(5, nwalkers)
    order = np.argsort(lp_last)  # ascending
    print(f"worst {k} walkers (by last logprob):", order[:k].tolist())
    print(f"best  {k} walkers (by last logprob):", order[-k:].tolist())

    # --- Quick stationarity check: compare early vs late halves (per parameter) ---
    flat = chain.reshape(nsteps * nwalkers, ndim)  # includes all steps
    half = nsteps // 2
    flat1 = chain[:half].reshape(half * nwalkers, ndim)
    flat2 = chain[half:].reshape((nsteps - half) * nwalkers, ndim)

    mean1 = np.mean(flat1, axis=0)
    mean2 = np.mean(flat2, axis=0)
    std2  = np.std(flat2, axis=0) + 1e-30  # avoid /0
    z = np.abs(mean2 - mean1) / std2
    print(f"stationarity (|Δmean|/std in last half): median={np.median(z):.2f}, max={np.max(z):.2f}")

    # --- Autocorr time / ESS (use only last half to avoid transient; may fail) ---
    try:
        tau = sampler.get_autocorr_time(tol=0)  # ndarray (ndim,)
        # Effective samples ~ (nsteps*nwalkers) / tau
        ess = (nsteps * nwalkers) / tau
        print("autocorr time tau: median={:.1f}, max={:.1f}".format(np.median(tau), np.max(tau)))
        print("ESS: median={:.0f}, min={:.0f}".format(np.median(ess), np.min(ess)))
        # simple rule of thumb
        print("N/tau: median={:.1f}, min={:.1f}".format(np.median(nsteps / tau), np.min(nsteps / tau)))
    except Exception as e:
        print(f"autocorr/ESS: skipped ({type(e).__name__}: {e})")

    print("===== end summary =====\n")

    return chain, logprob, sampler