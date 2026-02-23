"""
Warren & Dye (2003) の一番簡単な Semi-Linear Inversion (SLI) の実装
- lens fix, source grid fix, regularizationなし

要は、image planeの pixel ごとに、source plane の pixel からの contribution を計算して、前方行列を構築する。
As = d となるAを作れば、あとは、s = (A^T A)^{-1} A^T d で source plane の pixel の brightness を求めることができる。
"""

import numpy as np
from scipy.sparse import csr_matrix


import numpy as np
from scipy.sparse import csr_matrix

def build_matrix_lensing(
    beta_x_arcsec: np.ndarray,
    beta_y_arcsec: np.ndarray,
    mask: np.ndarray,
    ctx,
) -> csr_matrix:
    """
    lensing matrix L を作る（bilinear interpolation の係数を sparse 行列にする）

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
    # これが L の非ゼロ要素になる（各行最大4つ）
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
    Nvalid = rows.size

    # source pixel の「flatten index」を作る
    # flatten規約：index = j*nx + i  （yが先、xが後）
    col00 = (j0     * nx_src + (i0    )).astype(np.int64)
    col10 = (j0     * nx_src + (i0 + 1)).astype(np.int64)
    col01 = ((j0+1) * nx_src + (i0    )).astype(np.int64)
    col11 = ((j0+1) * nx_src + (i0 + 1)).astype(np.int64)

    # csr_matrix 用に、(row, col, data) を全部並べる
    # 各行4要素なので、長さは 4*Nvalid
    row_idx = np.repeat(rows, 4)  # [r,r,r,r, r,r,r,r, ...]
    col_idx = np.concatenate([col00, col10, col01, col11])
    data    = np.concatenate([w00,  w10,  w01,  w11 ]).astype(np.float32)

    # sparse 行列を作る
    # 形状は (Nimg_used, Nsrc) のままでOK
    # inside=False の行は “全部ゼロ行” になる（=寄与なし）
    L = csr_matrix((data, (row_idx, col_idx)), shape=(Nimg_used, Nsrc))

    return L





def build_forward_matrix(lens_model, source_grid, ctx):
    """
    A = B * L (B: beam convolution, L: lensing)に基づいて前方行列を構築する。
    後で uv-plane modelingをする際には、A = F * B * L (F: FFT, B: beam convolution, L: lensing)に拡張
    """
    # lensing
    A_lens = lens_model.lensing_matrix(source_grid, ctx)

    # beam convolution
    A_beam = lens_model.beam_convolution_matrix(ctx)

    # 前方行列
    A = A_beam @ A_lens

    return A