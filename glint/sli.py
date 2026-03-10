"""
Warren & Dye (2003) の一番簡単な Semi-Linear Inversion (SLI) の実装
- lens fix, source grid fix, regularizationなし

要は、image planeの pixel ごとに、source plane の pixel からの contribution を計算して、前方行列を構築する。
As = d となるAを作れば、あとは、s = (A^T A)^{-1} A^T d で source plane の pixel の brightness を求めることができる。
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import solve


def build_matrix_lensing(
    beta_x_arcsec: np.ndarray,
    beta_y_arcsec: np.ndarray,
    mask: np.ndarray,
    ctx,
) -> csr_matrix:
    """
    lensing matrix L を作る。すなわちbilinear interpolation の係数を sparse 行列にする。

    I_img(masked) = L @ S_src(flat)

    Parameters
    ----------
    beta_x_arcsec, beta_y_arcsec : (ny_img, nx_img)
        image plane各ピクセルに対応する source plane座標（arcsec）
    mask : (ny_img, nx_img) bool
        True の image pixel だけ使う
    ctx : ImageContext
        必要: ctx.pixsize_src, ctx.x0_src, ctx.y0_src, ctx.xx_src.shape

    Returns
    -------
    L : csr_matrix  shape (Nimg_used, Nsrc)
        image の1ピクセルが、source のどの4ピクセルの線形結合か
    """

    # Source plane
    ny_src, nx_src = ctx.xx_src.shape
    Nsrc = ny_src * nx_src

    # Source center pixel indices
    x0_src_pix = (nx_src - 1) / 2.0
    y0_src_pix = (ny_src - 1) / 2.0

    # maskされた image pixel だけ取り出す
    mask_flat = mask.ravel()
    bx = beta_x_arcsec.ravel()[mask_flat]   # (Nimg_used,)
    by = beta_y_arcsec.ravel()[mask_flat]   # (Nimg_used,)
    Nimg_used = bx.size

    # source plane 座標 (arcsec) -> source plane の pixel index に変換
    bx_pix = (bx - ctx.x0_src) / ctx.pixsize_src + x0_src_pix  # float pixel coordinate
    by_pix = (by - ctx.y0_src) / ctx.pixsize_src + y0_src_pix  # float pixel coordinate

    # ここからmap_source_to_imageでmap_coordinatesの中身と同じことをやる
    # bilinear interpolation の「左下」ピクセル index を求める
    #     i0 = floor(x), j0 = floor(y)
    i0 = np.floor(bx_pix).astype(np.int64)  # x方向の整数index
    j0 = np.floor(by_pix).astype(np.int64)  # y方向の整数index

    # 小数部分 dx, dy を求める（0〜1）
    # 4点補間の重みの元
    dx = bx_pix - i0
    dy = by_pix - j0

    # 4近傍 (i0,j0), (i0+1,j0), (i0,j0+1), (i0+1,j0+1) の重み
    # これが L の非ゼロ要素になる（各行4つ）
    w00 = (1 - dx) * (1 - dy)
    w10 = (dx)     * (1 - dy)
    w01 = (1 - dx) * (dy)
    w11 = (dx)     * (dy)

    # 境界の外に出たやつは寄与ゼロ（= 行全体ゼロ）にしたい
    # なので4近傍が全部source画像内にある点だけ残す
    inside = (
        (i0 >= 0) & (i0 + 1 < nx_src) &
        (j0 >= 0) & (j0 + 1 < ny_src)
    )

    # inside でフィルタして、行番号も作り直す
    rows = np.nonzero(inside)[0]  # (Nvalid,)   0..Nimg_used-1 のうち有効な行
    i0 = i0[inside]; j0 = j0[inside]
    w00 = w00[inside]; w10 = w10[inside]; w01 = w01[inside]; w11 = w11[inside]

    # 有効な行数
    # Nvalid = rows.size

    # source pixel の「flatten index」を作る
    # flatten規約：index = j*nx + i  （yが先、xが後）
    col00 = (j0     * nx_src + (i0    )).astype(np.int64)
    col10 = (j0     * nx_src + (i0 + 1)).astype(np.int64)
    col01 = ((j0+1) * nx_src + (i0    )).astype(np.int64)
    col11 = ((j0+1) * nx_src + (i0 + 1)).astype(np.int64)

    # csr_matrix 用に、(row, col, data) を全部並べる
    # 各行最大4要素なので、長さは 4*Nvalid
    # row_idx = np.repeat(rows, 4)  # [r,r,r,r, r,r,r,r, ...]
    row_idx = np.concatenate([rows, rows, rows, rows])
    col_idx = np.concatenate([col00, col10, col01, col11])
    data    = np.concatenate([w00,  w10,  w01,  w11]).astype(np.float64)
    # sparse 行列を作る
    # 形状は (Nimg_used, Nsrc) のままでOK
    # inside=False の行は “全部ゼロ行” になる（=寄与なし）
    L = csr_matrix((data, (row_idx, col_idx)), shape=(Nimg_used, Nsrc))

    return L


def mask_source_supported_pixels(
    beta_x_arcsec: np.ndarray,
    beta_y_arcsec: np.ndarray,
    mask: np.ndarray,
    ctx,
) -> np.ndarray:
    """
    Keep only image pixels whose 4-neighbor bilinear stencil is fully inside source grid.
    """
    ny_src, nx_src = ctx.xx_src.shape
    x0_src_pix = (nx_src - 1) / 2.0
    y0_src_pix = (ny_src - 1) / 2.0

    bx_pix = (beta_x_arcsec - ctx.x0_src) / ctx.pixsize_src + x0_src_pix
    by_pix = (beta_y_arcsec - ctx.y0_src) / ctx.pixsize_src + y0_src_pix

    i0 = np.floor(bx_pix).astype(np.int64)
    j0 = np.floor(by_pix).astype(np.int64)

    inside = (
        (i0 >= 0) & (i0 + 1 < nx_src) &
        (j0 >= 0) & (j0 + 1 < ny_src)
    )

    return mask & inside

# def build_forward_matrix(lens_model, source_grid, ctx):
#     """
#     A = B * L (B: beam convolution, L: lensing)に基づいて前方行列を構築する。
#     後で uv-plane modelingをする際には、A = F * B * L (F: FFT, B: beam convolution, L: lensing)に拡張
#     """
#     # lensing
#     A_lens = lens_model.lensing_matrix(source_grid, ctx)

#     # beam convolution
#     A_beam = lens_model.beam_convolution_matrix(ctx)

#     # 前方行列
#     A = A_beam @ A_lens

#     return A

def extract_masked_vector(arr2d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flatten 2D array using the given boolean mask."""
    return np.asarray(arr2d, dtype=np.float64).ravel()[mask.ravel()]


def solve_sli(
    L: csr_matrix,
    d: np.ndarray,
    sigma: np.ndarray,
    lam: float = 0.0,
    H: np.ndarray | None = None,
    rcond: float = 1e-12,
):
    """
    Solve weighted least squares SLI:
        s = argmin ||(d - Ls)/sigma||^2  [+ lam * s^T H s]

    Parameters
    ----------
    L : csr_matrix, shape (Ndata, Nsrc)
    d : (Ndata,)
        Masked data vector.
    sigma : (Ndata,)
        1-sigma noise for each used image pixel.
    lam : float
        Regularization strength. For Phase 1, keep lam=0.
    H : (Nsrc, Nsrc) or None
        Regularization matrix. For Phase 1, keep H=None.
    rcond : float
        Small diagonal jitter for numerical stability.

    Returns
    -------
    result : dict
        's'        : best-fit source vector
        'F'        : normal matrix
        'D'        : RHS vector
        'cov'      : inverse(F)
        'model'    : L @ s
        'chi2'     : chi-square
        'ndof'     : degrees of freedom
    """
    d = np.asarray(d, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if d.ndim != 1 or sigma.ndim != 1:
        raise ValueError("d and sigma must be 1D arrays.")
    if d.size != L.shape[0] or sigma.size != L.shape[0]:
        raise ValueError("Size mismatch among L, d, sigma.")
    if np.any(~np.isfinite(d)) or np.any(~np.isfinite(sigma)):
        raise ValueError("d and sigma must be finite.")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be > 0.")

    w = 1.0 / sigma**2

    # Weighted normal equation
    LW = L.multiply(w[:, None])        # = W @ L
    F = (L.T @ LW).toarray()           # = L^T W L
    D = L.T @ (w * d)                  # = L^T W d
    D = np.asarray(D).reshape(-1)

    if H is not None and lam != 0.0:
        F = F + lam * np.asarray(H, dtype=np.float64)

    # small jitter
    F = F + rcond * np.eye(F.shape[0])

    s = solve(F, D, assume_a="sym")
    model = np.asarray(L @ s).reshape(-1)
    resid = (d - model) / sigma
    chi2 = float(np.sum(resid**2))

    cov = np.linalg.inv(F)
    ndof = d.size - s.size

    return {
        "s": s,
        "F": F,
        "D": D,
        "cov": cov,
        "model": model,
        "chi2": chi2,
        "ndof": ndof,
    }


def source_vector_to_2d(s: np.ndarray, ctx) -> np.ndarray:
    """Reshape flat source vector to 2D source image."""
    ny_src, nx_src = ctx.xx_src.shape
    return np.asarray(s).reshape(ny_src, nx_src)


def predict_image_from_source(L: csr_matrix, s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Put masked model vector back onto a full 2D image.
    Pixels outside mask are set to 0.
    """
    model_masked = np.asarray(L @ s).reshape(-1)
    out = np.zeros(mask.size, dtype=np.float64)
    out[mask.ravel()] = model_masked
    return out.reshape(mask.shape)


def residual_image(data2d: np.ndarray, model2d: np.ndarray, sigma2d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Residual image in sigma units on masked pixels, zero elsewhere.
    """
    out = np.zeros_like(data2d, dtype=np.float64)
    good = mask & np.isfinite(data2d) & np.isfinite(model2d) & np.isfinite(sigma2d) & (sigma2d > 0)
    out[good] = (data2d[good] - model2d[good]) / sigma2d[good]
    return out