"""
Image <--> Visibility transforming functions
"""

import numpy as np
# import numba as nb
from finufft import nufft2d1, nufft2d2, nufft2d3

ARCSEC2RAD = np.deg2rad(1/3600)


# def primary_beam(xx_as, yy_as, pb_fwhm_as):
#     r_as = np.hypot(xx_as, yy_as)
#     PB = np.exp(-4.0*np.log(2.0) * (r_as**2) / (pb_fwhm_as**2)) # ALMA technical handbook Fig 7.14
#     return PB

def primary_beam(xx_as, yy_as, pb_fwhm_as):
    r_as = np.hypot(xx_as, yy_as)
    pb_fwhm_as = np.asarray(pb_fwhm_as)
    # Support scalar, per-channel 1D, or full 2D PB FWHM inputs.
    if pb_fwhm_as.ndim == 0:
        denom = pb_fwhm_as**2
        PB = np.exp(-4.0*np.log(2.0) * (r_as**2) / denom)  # ALMA technical handbook Fig 7.14
        return PB
    if pb_fwhm_as.ndim == 1:
        # (nchan, 1, 1) broadcast with (ny, nx) -> (nchan, ny, nx)
        denom = (pb_fwhm_as[:, None, None] ** 2)
        PB = np.exp(-4.0*np.log(2.0) * (r_as[None, ...]**2) / denom)
        return PB
    if pb_fwhm_as.shape == r_as.shape:
        denom = pb_fwhm_as**2
        PB = np.exp(-4.0*np.log(2.0) * (r_as**2) / denom)
        return PB
    raise ValueError(
        "pb_fwhm_as must be scalar, 1D (nchan), or same shape as xx_as/yy_as. "
        f"Got pb_fwhm_as shape {pb_fwhm_as.shape} vs r_as shape {r_as.shape}."
    )


def image_to_vis_finufft_type2(
    I: np.ndarray,                 # 2D image [Jy/pixel]
    ps_arcsec: float,              # pixel size [arcsec]
    u: np.ndarray,                 # 1D CASA/MS u coordinates [wavelengths]
    v: np.ndarray,                 # 1D CASA/MS v coordinates [wavelengths]
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute image visibilities on a uniform grid with a type-2 NUFFT (uniform grid to non-uniform grid; much faster than type-3).
    https://finufft.readthedocs.io/en/latest/math.html
    
    The general type-2 NUFFT used here is
        c_j = sum_k f_k exp[i (s_k x_j + t_k y_j)]  (c_j: NU, f_k: U)
    
    The interferometric measurement equation (from ALMA technical handbook) is
        V(u,v) = integral I(l,m) exp[+2 pi i (u l + v m)] dl dm

    For the grid made by ``make_grid_arcsec`` (``x=-l``, ``y=m``), set
    ``x=-2 pi u dx`` and ``y=+2 pi v dy``. The image center must be the
    phase center.

    Parameters
    ----------
    I : 2D array
        Flux per pixel [Jy/pixel]. No normalization is applied.
    ps_arcsec : float
        Pixel size [arcsec] for scaling the NUFFT coordinates.
    u, v : 1D arrays
        CASA/MS coordinates [wavelengths].
    eps : float, optional
        NUFFT accuracy.

    Returns
    -------
    V : 1D complex array
        Complex visibilities.
    """
    ps_rad = ps_arcsec * ARCSEC2RAD
    x = -2.0 * np.pi * np.asarray(u, dtype=np.float64) * ps_rad
    y = +2.0 * np.pi * np.asarray(v, dtype=np.float64) * ps_rad

    # 画像は I = I[y, x] だが、FINUFFTは nufft2d2(x, y) の入力順であることに注意
    # I.T を渡しても良いが、転置のための余分なメモリコピーが発生する（ただし画像のfloat --> complexの変換のためにいずれにせよメモリコピーは必要でそこまで早くはならないが。実測すると FT 全体の1%程度）
    coefficients = np.asarray(I, dtype=np.complex128, order="C")
    return nufft2d2(x=y, y=x, f=coefficients, isign=+1, eps=eps)


def image_to_vis_finufft_type3(
    I: np.ndarray,                # 2D image [Jy/pixel]
    xx_as: np.ndarray,            # 2D image-plane x grid [arcsec]
    yy_as: np.ndarray,            # 2D image-plane y grid [arcsec]
    u: np.ndarray,                # 1D CASA/MS u coordinates [wavelengths]
    v: np.ndarray,                # 1D CASA/MS v coordinates [wavelengths]
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute image visibilities with a type-3 NUFFT (much slower but more general than type-2).

    The general type-3 NUFFT used here is
        f_k = sum_j c_j exp[i (s_k x_j + t_k y_j)]

    The interferometric measurement equation (from ALMA technical handbook) is
        V(u,v) = integral I(l,m) exp[+2 pi i (u l + v m)] dl dm

    Since make_grid_arcsec() in lensing.py defines ``x=-l`` and ``y=m``,
    this gives ``s=kx=-2 pi u`` and ``t=ky=+2 pi v``.
    Because the pixel coordinates are supplied explicitly, no ``fftshift`` is needed, and the grid may be nonuniform.

    This helper is intended for simple or one-off calculations.
    Repeated evaluations during sampling use a precomputed FINUFFT plan instead.

    Parameters
    ----------
    I : 2D array
        Flux per pixel [Jy/pixel]. No normalization is applied.
    xx_as, yy_as : 2D arrays
        Image coordinates [arcsec], with the phase center at zero.
    u, v : 1D arrays
        CASA/MS coordinates [wavelengths].
    eps : float, optional
        NUFFT accuracy.

    Returns
    -------
    V : 1D complex array
        Complex visibilities.
    """

    xx_rad = (xx_as * ARCSEC2RAD).ravel()
    yy_rad = (yy_as * ARCSEC2RAD).ravel()  # [rad]
    cj = I.ravel().astype(np.complex128)

    kx = -2.0 * np.pi * u
    ky = +2.0 * np.pi * v
    
    V = nufft2d3(x=xx_rad, y=yy_rad, c=cj, s=kx, t=ky, isign=+1, eps=eps)
    return V




# @nb.njit(parallel=True, fastmath=True)
# def nudft_numba(l, m, I, u, v):
#     """
#     l, m, I : (Npix,)  [rad, rad, Jy/pix]
#     u, v    : (Nvis,)  [wavelengths]
#     return  : (Nvis,) complex64
#     """
#     NV = u.shape[0]
#     NP = l.shape[0]
#     out = np.empty(NV, dtype=np.complex64)

#     for i in nb.prange(NV):
#         ui = u[i]
#         vi = v[i]
#         s_re = 0.0   # float64 でもOKだが、Iがfloat32なら 0.0 を np.float32(0.0) にしてもよい
#         s_im = 0.0
#         for j in range(NP):
#             ph = -2.0*np.pi*(ui*l[j] + vi*m[j])   # スカラー
#             c = np.cos(ph)                        # スカラー
#             s = np.sin(ph)                        # スカラー
#             s_re += I[j] * c                      # I[j] はスカラー
#             s_im += I[j] * s
#         out[i] = np.complex64(s_re + 1j*s_im)
#     return out



def vis_to_image_finufft_type1(
    u: np.ndarray,
    v: np.ndarray,
    V: np.ndarray,
    V_weight: np.ndarray,
    image_shape: tuple[int, int],
    ps_arcsec: float,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a dirty image on a uniform grid with a type-1 NUFFT.

    Type-1 evaluates
        ``f_k = sum_j c_j exp[-i (s_k x_j + t_k y_j)]``.

    The interferometric inverse transform is
        ``I(l,m) = Re[sum_j w_j V_j exp[-2 pi i (u_j l + v_j m)]] / sum_j w_j``.

    Parameters
    ----------
    u, v : 1D arrays
        CASA/MS coordinates [wavelengths].
    V : 1D complex array
        Visibilities [Jy].
    V_weight : 1D array
        Visibility weights.
    image_shape : tuple
        Output image shape ``(ny, nx)``.
    ps_arcsec : float
        Pixel size [arcsec].
    eps : float, optional
        NUFFT accuracy.

    Returns
    -------
    image, beam : 2D arrays
        Real dirty image and dirty beam, normalized by ``sum(V_weight)``.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    visibility = np.asarray(V, dtype=np.complex128)
    weight = np.asarray(V_weight, dtype=np.float64)
    if not (u.shape == v.shape == visibility.shape == weight.shape):
        raise ValueError("u, v, V, and V_weight must have the same shape.")
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError("image_shape must be (ny, nx) with positive sizes.")
    if ps_arcsec <= 0:
        raise ValueError("ps_arcsec must be > 0.")

    ny, nx = (int(value) for value in image_shape)
    if nx % 2 or ny % 2:
        raise ValueError(
            "type-1 imaging requires even image dimensions for the "
            "make_grid_arcsec center convention."
        )
    weight_sum = np.sum(weight)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("V_weight must have a positive finite sum.")

    ps_rad = ps_arcsec * ARCSEC2RAD
    x = -2.0 * np.pi * u * ps_rad
    y = +2.0 * np.pi * v * ps_rad
    modes = (nx, ny)  # 画像は I = I[y, x] だが、FINUFFTは (x, y) の順であることに注意

    image_xy = nufft2d1(
        x=x, y=y, c=visibility * weight, n_modes=modes,
        isign=-1, eps=eps,
    )
    beam_xy = nufft2d1(
        x=x, y=y, c=weight.astype(np.complex128), n_modes=modes,
        isign=-1, eps=eps,
    )
    return (image_xy.T / weight_sum).real, (beam_xy.T / weight_sum).real


def vis_to_image_finufft_type3(
    u: np.ndarray,
    v: np.ndarray,
    V: np.ndarray,
    V_weight: np.ndarray,
    xx_as: np.ndarray,
    yy_as: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a dirty image on explicit coordinates with a type-3 NUFFT.

    Type-3 evaluates
        ``f_k = sum_j c_j exp[-i (s_k x_j + t_k y_j)]``.

    The interferometric inverse transform is
        ``I(l,m) = Re[sum_j w_j V_j exp[-2 pi i (u_j l + v_j m)]] / sum_j w_j``.

    Parameters
    ----------
    u, v : 1D arrays
        CASA/MS coordinates [wavelengths].
    V : 1D complex array
        Visibilities [Jy].
    V_weight : 1D array
        Visibility weights.
    xx_as, yy_as : 2D arrays
        Image coordinates [arcsec], with the phase center at zero.
    eps : float, optional
        NUFFT accuracy.

    Returns
    -------
    image, beam : 2D arrays
        Real dirty image and dirty beam, normalized by ``sum(V_weight)``.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    visibility = np.asarray(V, dtype=np.complex128)
    weight = np.asarray(V_weight, dtype=np.float64)
    xx_as = np.asarray(xx_as, dtype=np.float64)
    yy_as = np.asarray(yy_as, dtype=np.float64)
    if not (u.shape == v.shape == visibility.shape == weight.shape):
        raise ValueError("u, v, V, and V_weight must have the same shape.")
    if xx_as.shape != yy_as.shape or xx_as.ndim != 2:
        raise ValueError("xx_as and yy_as must be 2D arrays with the same shape.")
    weight_sum = np.sum(weight)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("V_weight must have a positive finite sum.")

    kx = -2.0 * np.pi * u
    ky = +2.0 * np.pi * v
    x = (xx_as * ARCSEC2RAD).ravel()
    y = (yy_as * ARCSEC2RAD).ravel()

    image = nufft2d3(
        x=kx, y=ky, c=visibility * weight, s=x, t=y,
        isign=-1, eps=eps,
    ).reshape(xx_as.shape)
    beam = nufft2d3(
        x=kx, y=ky, c=weight.astype(np.complex128), s=x, t=y,
        isign=-1, eps=eps,
    ).reshape(xx_as.shape)
    return (image / weight_sum).real, (beam / weight_sum).real
