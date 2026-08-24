"""
residual / logprob functionは個々のnotebookに書く
"""

from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
import multiprocessing as mp
from scipy.optimize import least_squares
import emcee
import dynesty

from .pressure_support import UnphysicalPressureSupportError


# ------------------- fitting utilities -------------------
def run_least_squares(
        residual_fn, x0, lb, ub, *, method="trf", x_scale="jac",
        ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=1000, verbose=0):
    """
    residual_fn(theta) は、観測データとモデルの差を返すベクトル関数（例: (obs - model) / noise）
    method = "trf" (default): 境界付き最適化用
    x_scale = "jac": パラメータのスケールをヤコビアンのスケールに合わせる
    ftol: 目的関数の許容誤差
    xtol: パラメータの許容誤差
    gtol: 勾配の許容誤差
    """
    x0 = np.asarray(x0, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if x0.ndim != 1 or lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("x0, lb, and ub must be one-dimensional arrays with equal shape.")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("x0, lb, and ub must be finite.")
    if np.any(lb >= ub):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    if np.any((x0 < lb) | (x0 > ub)):
        raise ValueError("x0 must lie within the parameter bounds.")
    return least_squares(
        residual_fn,
        x0=x0,
        bounds=(lb, ub),
        method=method,
        x_scale=x_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=verbose,
    )


def run_parallel_fits(fit_fn, cases, *, max_workers=None):
    """独立した複数ケースを並列fitし、入力と同じ順序で結果を返す。

    ``cases`` は各ケースの設定を格納したiterableで、通常は
    ``{"name": ..., "data": ...}`` のようなdictのlistを想定する。
    ``fit_fn`` はその1要素を受け取り、ケース固有のresidualを定義して
    内部で ``run_least_squares`` を実行し、fit結果を返す関数。
    """
    cases = tuple(cases)
    if not cases:
        return []
    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        max_workers = cpu_count
    max_workers = min(len(cases), int(max_workers), cpu_count)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(fit_fn, cases))



# ------------------- dynesty utilities -------------------
def run_dynesty(
        loglikelihood, prior_transform, ndim, *, nlive=500, dlogz=0.1,
        bound="multi", sample="rwalk", rstate=None, print_progress=True):
    """dynestyによるnested samplingを実行し、sampling結果を返す。

    ``loglikelihood(theta)`` と、単位cubeをparameterへ変換する
    ``prior_transform(unit_cube)`` はデータごとに呼び出し側で定義する。
    """
    if int(ndim) < 1:
        raise ValueError("ndim must be >= 1.")
    sampler = dynesty.NestedSampler(
        loglikelihood,
        prior_transform,
        int(ndim),
        nlive=int(nlive),
        bound=bound,
        sample=sample,
        rstate=rstate,
    )
    sampler.run_nested(dlogz=float(dlogz), print_progress=print_progress)
    return sampler.results


def run_dynamic_dynesty(
        loglikelihood, prior_transform, ndim, *, nlive_init=500,
        nlive_batch=500, dlogz_init=0.1, pfrac=0.5,
        evidence_threshold=0.1, target_n_effective=10000,
        max_batches=None, bound="multi", sample="rwalk", rstate=None,
        pool=None, queue_size=None, use_pool=None, print_progress=True):
    """DynamicNestedSamplerを実行し、sampling結果を返す。

    ``pfrac`` は追加samplingをposteriorへ割り当てる割合で、0なら
    evidenceのみ、1ならposteriorのみを優先する。
    ``pool`` と ``queue_size`` を指定するとlikelihood評価を並列化できる。
    """
    if dynesty is None:
        raise ImportError("run_dynamic_dynesty requires the optional 'dynesty' package.")
    if int(ndim) < 1:
        raise ValueError("ndim must be >= 1.")
    if not 0.0 <= float(pfrac) <= 1.0:
        raise ValueError("pfrac must be between 0 and 1.")
    sampler = dynesty.DynamicNestedSampler(
        loglikelihood,
        prior_transform,
        int(ndim),
        nlive=int(nlive_init),
        bound=bound,
        sample=sample,
        rstate=rstate,
        pool=pool,
        queue_size=queue_size,
        use_pool=use_pool,
    )
    sampler.run_nested(
        nlive_init=int(nlive_init),
        nlive_batch=int(nlive_batch),
        dlogz_init=float(dlogz_init),
        wt_kwargs={"pfrac": float(pfrac)},
        stop_kwargs={
            "pfrac": float(pfrac),
            "evid_thresh": float(evidence_threshold),
            "target_n_effective": int(target_n_effective),
        },
        maxbatch=None if max_batches is None else int(max_batches),
        print_progress=print_progress,
    )
    return sampler.results



# ------------------- MCMC utilities -------------------
LogProbFn = Callable[[np.ndarray], float]

_FORKED_LOGPROB_FN: Optional[LogProbFn] = None


def _evaluate_forked_logprob(theta):
    if _FORKED_LOGPROB_FN is None:
        raise RuntimeError("Parallel log-probability worker was not initialized.")
    return _FORKED_LOGPROB_FN(theta)


def logprior_box(theta, lb, ub):
    """
    ステップ状の事前分布
    """
    if np.any(theta < lb) or np.any(theta > ub):
        return -np.inf
    return 0.0


def make_initial_walkers(x0, lb, ub, nwalkers, seed=0, init_frac=0.2, max_tries=10000000):
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
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError(
                "Parallel run_emcee requires the multiprocessing 'fork' start method "
                "when logprob_fn is defined inside a notebook function."
            )

        global _FORKED_LOGPROB_FN
        _FORKED_LOGPROB_FN = logprob_fn
        try:
            with mp.get_context("fork").Pool(processes=ncpu) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    _evaluate_forked_logprob,
                    pool=pool,
                )

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
        finally:
            _FORKED_LOGPROB_FN = None

    
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
    # max - median が（桁が違うくらい）全然違うと、一部が外れている可能性高い
    print(f"logprob (last step): finite fraction={np.mean(finite_last):.3f}, "
          f"median={np.median(lp_last[finite_last]):.3f}, max={np.max(lp_last[finite_last]):.3f}")

    # show worst walkers by last-step logprob (useful for diagnosing stuck/outliers)
    k = min(5, nwalkers)
    order = np.argsort(lp_last)  # ascending
    print(f"worst {k} walkers (by last logprob):", order[:k].tolist())
    print(f"best  {k} walkers (by last logprob):", order[-k:].tolist())

    # --- Quick stationarity check: compare early vs late halves (per parameter) 要は前半と後半で分布がずれていないか ---
    flat = chain.reshape(nsteps * nwalkers, ndim)  # includes all steps
    half = nsteps // 2
    flat1 = chain[:half].reshape(half * nwalkers, ndim)
    flat2 = chain[half:].reshape((nsteps - half) * nwalkers, ndim)

    mean1 = np.mean(flat1, axis=0)
    mean2 = np.mean(flat2, axis=0)
    std2  = np.std(flat2, axis=0) + 1e-30  # avoid /0
    z = np.abs(mean2 - mean1) / std2
    # median<0.3 & max<1くらいならまあ同じ分布からのサンプルっぽい。max>3とかだと一部のパラメータが全然収束してない可能性がある。
    print(f"stationarity (|Δmean|/std in last half): median={np.median(z):.2f}, max={np.max(z):.2f}")

    # --- Autocorr time / ESS (use only last half to avoid transient; may fail) ---
    # tauは独立サンプルになるまでのステップ数、小さいほど嬉しいが、縮退が強いと大きくなりがち
    # N/tau: >50だと安心、>20くらいはほしい。<10だとあまり信用できない
    # ESS: >500くらいあればまあOK、>1000くらいあると安心。>2000だと素晴らしい、<200だと危険（誤差がぶれやすい）
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


# ------------------- likelihood factories -------------------
def make_visibility_loglikelihood(model, layout, uv_context, *, scale=1.0):
    """複素visibilityのGaussian log-likelihoodを返すfactory
    visibility計算には C++のfinufft.Planを使用している。これは固定値だが、plan.execute(coeff)を実行するとcoeffに応じて内部のscratchバッファに書き込むのでstatefulになる。
    GILはあくまでPythonオブジェクトに限られ、C++の内部状態は保護されないので注意
    だから、各threadに専用のloglikelihood（専用UVContext/planを内包する closure）を割り当てる必要がある

    Parameters
    ----------
    model : ForwardModel
    layout : ParameterLayout
    uv_context : data/sigma入りのUVContext（``build_uv_observation``の出力）
    scale : mock観測のflux_scaleなど、model cubeへ掛ける定数
    """

    data = uv_context.data
    sigma = uv_context.sigma
    uv_ctx = uv_context
    log_normalization = 2.0 * np.sum(np.log(2.0 * np.pi * sigma**2))  # 実部と虚部の両方を考慮するために2倍（evidenceの絶対値に入るので）

    def loglikelihood(theta):
        try:
            visibilities = model.make_visibilities(layout.decode(theta), uv_ctx, scale=scale)
        except UnphysicalPressureSupportError: # pressure supportが重力を上回る非物理なsampleは-infで棄却
            return -np.inf
        chi2 = np.sum(np.abs((data - visibilities) / sigma) ** 2)
        return -0.5 * (chi2 + log_normalization)

    return loglikelihood


def make_image_loglikelihood(model, layout, data_cube, noise_sigma, mask, *,
                             scale=1.0, primary_beam=None):
    """image-domain（beam畳み込み後cube）のGaussian log-likelihoodを返すfactory
    各voxelのnoiseが独立で、既知の同じsigmaを持つと仮定している（dirty imageに使う場合はpixel相関を無視した近似）
    （こっちはstatefulではないので、各threadで同じloglikelihoodを使っても問題ない）
    
    Parameters
    ----------
    model : ForwardModel（beam必須）
    layout : ParameterLayout
    data_cube : (nchan, ny, nx) 観測cube
    noise_sigma : voxelあたりのnoise標準偏差（scalar）
    mask : (nchan, ny, nx) bool。Trueのvoxelだけfitする
    """

    data = np.asarray(data_cube)[mask]
    log_normalization = data.size * np.log(2.0 * np.pi * noise_sigma**2)

    def loglikelihood(theta):
        try:
            model_cube = model.make_convolved_lensed_cube(
                layout.decode(theta), scale=scale, primary_beam=primary_beam,
            )[mask]
        except UnphysicalPressureSupportError:
            return -np.inf
        chi2 = np.sum(((data - model_cube) / noise_sigma) ** 2)
        return -0.5 * (chi2 + log_normalization)

    return loglikelihood


# ------------------- parallel dynesty -------------------
def run_dynamic_dynesty_parallel(
        make_loglikelihood, prior_transform, ndim, *, workers=1,
        print_progress=True, **dynesty_options):
    """1つのfitを複数threadで実行するrun_dynamic_dynestyのwrapper。

    FINUFFT Plan.execute()はstatefulなので、同じplanを複数threadで共有せず、各threadへ専用のloglikelihood（専用UVContext/planを内包する closure）を一つずつ割り当てる。
    ``make_loglikelihood()``は呼ぶたびに独立した資源を持つloglikelihoodを返すfactoryでないといけない（uv fitでは呼ぶたびにUVContextを再構築する）
    読み取り専用資源しか使わないimage-domain fitでは、同じclosureを返すfactoryでよい。

    workers=1では従来のserial実行と完全に同一
    """
    if int(workers) < 1:
        raise ValueError("workers must be >= 1.")
    if workers == 1:
        return run_dynamic_dynesty(
            make_loglikelihood(), prior_transform, ndim,
            print_progress=print_progress, **dynesty_options,
        )

    from multiprocessing.pool import ThreadPool
    from queue import SimpleQueue
    from threading import local

    # workerと同じ数だけ、独立したUVContext/planを持つloglikelihoodを先に作っておく
    functions = [make_loglikelihood() for _ in range(int(workers))]
    # SimpleQueueはthread-safeなので、各threadが排他的に1個だけ取り出せる
    function_queue = SimpleQueue()
    for function in functions:
        function_queue.put(function)
    # thread-local storage: 同じ変数名でも、OS threadごとに別の値を持てる。
    thread_state = local()

    def initialize_likelihood_thread():
        # ThreadPoolの各workerが起動した直後に1回だけ呼ばれるinitializer。
        # ここでqueueから1個取り出し、そのthread専用のthread_stateへ固定する。
        # 以後そのthreadはずっと同じclosure（＝同じPlanインスタンス）だけを使い他のthreadと資源を共有しない。
        thread_state.loglikelihood = function_queue.get()

    def loglikelihood(theta):
        # dynestyへ渡す関数は1個だが、実体は呼び出し元threadのthread_stateへ委譲するので、thread AとBが同時に呼んでも別々のPlanが動く。
        return thread_state.loglikelihood(theta)

    # processes=workersでthread数をfunctionsの数に一致させ、1 thread : 1 closureの対応を過不足なく成立させる。
    pool = ThreadPool(processes=int(workers), initializer=initialize_likelihood_thread)
    try:
        return run_dynamic_dynesty(
            loglikelihood, prior_transform, ndim,
            pool=pool, queue_size=int(workers),
            use_pool={
                # 重いloglikelihood評価とpropose_point（新live pointの提案）だけをpoolへ回す。prior_transform/update_boundは軽いのでmain threadのまま。
                'prior_transform': False, 'loglikelihood': True, 'propose_point': True, 'update_bound': False,
            },
            print_progress=print_progress, **dynesty_options,
        )
    finally:
        # 全workerがexecute()を終えるのを待ってからpoolを閉じる。ここで待たずに抜けるとPlanの後片付けと実行中のexecute()が競合し得る。
        pool.close()
        pool.join()


def weighted_quantile(values, quantiles, weights):
    """Nested samplingの不均一なposterior weightを考慮した分位点。"""
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)
