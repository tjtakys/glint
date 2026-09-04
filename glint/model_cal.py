"""アンテナ位相の自己較正（model_cal）: Hezaveh+13 App. A の複素残差版。

visilens model_cal functionを参考にした（https://github.com/jspilker/visilens/blob/master/visilens/calc_likelihood.py#L378）
visilensの model_cal は 位相差 ``angle(V_data) − angle(V_model)`` の線形最小二乗だが、輝線データの低per-visibility SNR
（G09で|V|/σ中央値≈1）では破綻する。ここでは複素残差

    χ²(φ) = Σ_k |V_data,k − V_model,k exp(i(φ_{g,a1} − φ_{g,a2}))|² / σ_k²  +  Σ φ² / σ_φ²

をGauss-Newtonで最小化する（self-calと同形。高SNR極限でH+13線形解に一致）。
φは群g（``solution_interval``: (dataset, scan) または dataset）ごとに独立で、群内の基準antenna
（local index 0）を0に固定する。全群を(n_group, A, A)のブロック対角Fisher行列でバッチsolveする。
補正はモデル側に掛ける（``apply_antenna_phases``）。データは触らない。

このmoduleは純NumPyで状態を持たないので、FINUFFT planのようなthread専用化は不要。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .context import flatten_selected


@dataclass(frozen=True, slots=True)
class PhaseCalLayout:
    """flatten済みvisibility（UVContextの ``slices`` 順）ごとの群・antenna index と、
    Fisher行列組立用の事前計算index。読み取り専用で全thread/processで共有できる。"""

    group: np.ndarray            # (Ntot,) int: 群番号 0..n_group-1
    ant1: np.ndarray             # (Ntot,) int: 群内local antenna index（0が基準antenna）
    ant2: np.ndarray             # (Ntot,) int
    n_group: int
    n_ant_max: int               # 群内antenna数の最大値 A（Fisher行列の辺）
    group_dataset: np.ndarray    # (n_group,) int: 群のdataset_index
    group_scan: np.ndarray       # (n_group,) int: 群のscan番号（solution_interval="dataset"では-1）
    group_antennas: tuple        # 群ごとの元antenna番号配列（local index → MS antenna id）
    dataset_names: tuple
    solution_interval: str
    # bincount用の平坦index（群gのlocal antenna a,b について (g*A+a)*A+b、右辺は g*A+a）
    index_f11: np.ndarray
    index_f22: np.ndarray
    index_f12: np.ndarray
    index_f21: np.ndarray
    index_b1: np.ndarray
    index_b2: np.ndarray

    @property
    def n_visibilities(self) -> int:
        return int(self.group.size)


@dataclass(frozen=True, slots=True)
class PhaseSolution:
    phi: np.ndarray          # (n_group, A) rad。基準antennaと群に現れないantennaは0
    sigma_phi: np.ndarray    # (n_group, A) rad。予測1σ sqrt(diag F⁻¹)。基準/未使用はnan
    used: np.ndarray         # (n_group, A) bool。群に現れるantenna（基準antennaも含む）
    logdet: float            # Σ_g log det(F_g σ_φ²)（priorなしでは Σ_g log det F_g）
    n_iterations: int
    converged: bool

    @property
    def n_parameters(self) -> int:
        """解いた位相の自由度（群ごとの基準antennaを除く）。"""
        return int(self.used.sum() - self.used.shape[0])


def build_phase_cal_layout(
    uv,
    flag: np.ndarray,
    slices: Sequence[Optional[slice]],
    *,
    solution_interval: str = "scan",
) -> PhaseCalLayout:
    """``ProcessedUVData`` の行metadata（ant1/ant2/scan/dataset_index）と、UVContextの
    ``flag``/``slices`` から ``PhaseCalLayout`` を作る。

    群はdataset境界を跨がない（antenna番号も位相誤差もMSごとに無関係）。
    ``solution_interval="scan"`` は (dataset, scan) ごと、``"dataset"`` はdatasetごとに1解。
    位相誤差は全channel共通なので、行ごとの群・antenna indexをchannel方向へbroadcastして
    ``flatten_selected`` で(Ntot,)へ展開する。
    """
    if uv.ant1 is None or uv.ant2 is None:
        raise ValueError("uv.ant1/ant2 are required for model_cal (old npz without antenna metadata).")
    if solution_interval not in ("scan", "dataset"):
        raise ValueError('solution_interval must be "scan" or "dataset".')
    flag = np.asarray(flag, dtype=bool)
    nchan, nrow = flag.shape
    ant1 = np.asarray(uv.ant1, dtype=np.int64)
    ant2 = np.asarray(uv.ant2, dtype=np.int64)
    if ant1.shape != (nrow,) or ant2.shape != (nrow,):
        raise ValueError(f"ant1/ant2 must have shape ({nrow},).")
    if np.any(ant1 == ant2):
        raise ValueError("auto-correlations (ant1 == ant2) are not supported.")

    dataset = (np.zeros(nrow, dtype=np.int64) if uv.dataset_index is None
               else np.asarray(uv.dataset_index, dtype=np.int64))
    dataset_names = (tuple(uv.dataset_names) if uv.dataset_names is not None
                     else tuple(f"dataset{i}" for i in range(int(dataset.max()) + 1)))
    if solution_interval == "scan":
        if uv.scan is None:
            raise ValueError('uv.scan is required for solution_interval="scan".')
        scan = np.asarray(uv.scan, dtype=np.int64)
        key = np.stack([dataset, scan], axis=1)
    else:
        scan = np.full(nrow, -1, dtype=np.int64)
        key = dataset[:, None]
    unique_keys, row_group = np.unique(key, axis=0, return_inverse=True)
    row_group = np.asarray(row_group).ravel()
    n_group = unique_keys.shape[0]

    # 群ごとにantenna集合を作り、local index（0が基準antenna=最小番号）へ写す。
    local1 = np.empty(nrow, dtype=np.int64)
    local2 = np.empty(nrow, dtype=np.int64)
    group_antennas = []
    for g in range(n_group):
        rows = row_group == g
        antennas = np.unique(np.concatenate([ant1[rows], ant2[rows]]))
        group_antennas.append(antennas)
        local1[rows] = np.searchsorted(antennas, ant1[rows])
        local2[rows] = np.searchsorted(antennas, ant2[rows])
    n_ant_max = max(a.size for a in group_antennas)

    def flat(rows_array):
        return flatten_selected(np.broadcast_to(rows_array[None, :], (nchan, nrow)), flag, slices)

    group = flat(row_group)
    a1 = flat(local1)
    a2 = flat(local2)
    A = n_ant_max
    base = group * A
    return PhaseCalLayout(
        group=group, ant1=a1, ant2=a2, n_group=int(n_group), n_ant_max=int(A),
        group_dataset=unique_keys[:, 0].astype(np.int64),
        group_scan=(unique_keys[:, 1].astype(np.int64) if solution_interval == "scan"
                    else np.full(n_group, -1, dtype=np.int64)),
        group_antennas=tuple(group_antennas), dataset_names=dataset_names,
        solution_interval=solution_interval,
        index_f11=(base + a1) * A + a1, index_f22=(base + a2) * A + a2,
        index_f12=(base + a1) * A + a2, index_f21=(base + a2) * A + a1,
        index_b1=base + a1, index_b2=base + a2,
    )


def _phase_difference(layout: PhaseCalLayout, phi: np.ndarray) -> np.ndarray:
    phi_flat = np.asarray(phi, dtype=float).ravel()
    return phi_flat[layout.index_b1] - phi_flat[layout.index_b2]


def apply_antenna_phases(vis: np.ndarray, layout: PhaseCalLayout, phi: np.ndarray) -> np.ndarray:
    """``vis · exp(i(φ_{g,a1} − φ_{g,a2}))`` を返す（モデルへの利得適用、または注入テスト用）。"""
    return np.asarray(vis) * np.exp(1j * _phase_difference(layout, phi))


def solve_antenna_phases(
    data: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
    layout: PhaseCalLayout,
    *,
    prior_sigma_rad: Optional[float] = np.deg2rad(20.0),
    max_iterations: int = 8,
    tolerance: float = 1e-6,
) -> PhaseSolution:
    """複素残差のGauss-Newtonでantenna位相を解く（PROJECT_NOTES §10.10 D1）。

    1反復: ``rot = model·exp(iΔφ)``、``w = |rot|²/σ²``、``c = Im(conj(rot)·data)/σ²``、
    ``F = A W Aᵀ + I/σ_φ²``、``b = A c − φ/σ_φ²``、``δφ = F⁻¹ b``。
    初期値はφ=0固定（尤度をθの決定的関数に保つため、warm startはしない）。
    ``prior_sigma_rad=None`` で事前分布なし（visilens相当の純最尤解）。
    """
    data = np.asarray(data, dtype=np.complex128)
    model = np.asarray(model, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = layout.n_visibilities
    if not (data.shape == model.shape == sigma.shape == (n,)):
        raise ValueError(f"data/model/sigma must have shape ({n},) matching the layout.")
    G, A = layout.n_group, layout.n_ant_max
    inv_var = 1.0 / sigma**2
    prior = None if prior_sigma_rad is None or not np.isfinite(prior_sigma_rad) else 1.0 / float(prior_sigma_rad) ** 2

    counts = (np.bincount(layout.index_b1, minlength=G * A)
              + np.bincount(layout.index_b2, minlength=G * A)).reshape(G, A)
    used = counts > 0
    unused_free = ~used[:, 1:]  # 基準antennaを除いた未使用antenna
    eye = np.eye(A - 1)

    phi = np.zeros((G, A))
    converged = False
    n_iterations = 0
    fisher = None
    for n_iterations in range(1, int(max_iterations) + 1):
        rot = model * np.exp(1j * _phase_difference(layout, phi))
        w = inv_var * (rot.real**2 + rot.imag**2)
        c = inv_var * (rot.real * data.imag - rot.imag * data.real)  # Im(conj(rot)·data)
        size = G * A * A
        fisher = (np.bincount(layout.index_f11, w, size) + np.bincount(layout.index_f22, w, size)
                  - np.bincount(layout.index_f12, w, size) - np.bincount(layout.index_f21, w, size)
                  ).reshape(G, A, A)[:, 1:, 1:]
        b = (np.bincount(layout.index_b1, c, G * A) - np.bincount(layout.index_b2, c, G * A)
             ).reshape(G, A)[:, 1:]
        if prior is not None:
            fisher = fisher + prior * eye
            b = b - prior * phi[:, 1:]
        else:
            # 未使用antennaは対角1・右辺0で解0にする（特異行列を避ける）。
            fisher = fisher + np.einsum("gi,ij->gij", unused_free.astype(float), eye)
        step = np.linalg.solve(fisher, b[..., None])[..., 0]
        step[unused_free] = 0.0
        phi[:, 1:] += step
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break

    fisher_inv = np.linalg.inv(fisher)
    sigma_phi = np.full((G, A), np.nan)
    sigma_phi[:, 1:] = np.sqrt(np.einsum("gii->gi", fisher_inv))
    sigma_phi[:, 1:][unused_free] = np.nan
    scaled = fisher / prior if prior is not None else fisher
    sign, logabsdet = np.linalg.slogdet(scaled)
    if np.any(sign <= 0):
        raise FloatingPointError("Fisher matrix is not positive definite.")
    return PhaseSolution(
        phi=phi, sigma_phi=sigma_phi, used=used, logdet=float(logabsdet.sum()),
        n_iterations=int(n_iterations), converged=bool(converged),
    )


def chi2_summary(data, model, sigma, layout: PhaseCalLayout, solution: PhaseSolution) -> dict:
    """位相補正前後のχ²と、自由度あたりの改善量を返す。

    位相誤差が無くnoiseだけなら、各自由度はχ²を約1ずつ下げるので
    ``delta_chi2_per_parameter ≈ 1`` が帰無仮説の期待値。≫1なら有意な位相構造がある。
    """
    data = np.asarray(data, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.float64)
    model = np.asarray(model, dtype=np.complex128)
    corrected = apply_antenna_phases(model, layout, solution.phi)
    chi2_before = float(np.sum(np.abs((data - model) / sigma) ** 2))
    chi2_after = float(np.sum(np.abs((data - corrected) / sigma) ** 2))
    n_parameters = solution.n_parameters
    return {
        "chi2_before": chi2_before, "chi2_after": chi2_after,
        "delta_chi2": chi2_before - chi2_after, "n_parameters": n_parameters,
        "delta_chi2_per_parameter": (chi2_before - chi2_after) / max(n_parameters, 1),
        "n_visibilities": int(data.size),
    }


def antenna_phase_table(layout: PhaseCalLayout, solution: PhaseSolution) -> dict:
    """群×antennaの位相解を平坦な表（dictの1D配列）にする。基準antennaと未使用antennaは除く。"""
    g, a = np.nonzero(solution.used & np.isfinite(solution.sigma_phi))
    phi = solution.phi[g, a]
    sig = solution.sigma_phi[g, a]
    return {
        "group": g,
        "dataset": np.array([layout.dataset_names[layout.group_dataset[i]] for i in g]),
        "scan": layout.group_scan[g],
        "antenna": np.array([layout.group_antennas[i][j] for i, j in zip(g, a)]),
        "phi_deg": np.rad2deg(phi),
        "sigma_deg": np.rad2deg(sig),
        "significance": phi / sig,
    }
