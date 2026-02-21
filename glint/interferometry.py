import numpy as np
# import numba as nb
from finufft import nufft2d2, nufft2d3

ARCSEC2RAD = np.deg2rad(1/3600)

def primary_beam(xx_as, yy_as, pb_fwhm_as):
    r_as = np.hypot(xx_as, yy_as)
    PB = np.exp(-4.0*np.log(2.0) * (r_as**2) / (pb_fwhm_as**2)) # ALMA technical handbook Fig 7.14
    return PB

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
    I,                 # 2D image [Jy/pix]
    ps_arcsec,        # pixel size [arcsec]
    u, v,              # 1D arrays [wavelengths]
    *,
    eps=1e-6           # NUFFT精度
):
    """
    画像 I(x,y) から 非等間隔(u,v)の視線データ V(u,v) を計算する（FFT + NUFFT type-2）
    ****画像の中心がphase centerに対応していることが前提****

    Parameters
    ----------
    I_img : 2D array
        Input image [Jy/pix] ** NOT BRIGHTNESS ** : PB correction済みとする
    ps_arcsec : float
        Pixel size [arcsec]
    u : 1D array
        U coordinates [wavelengths]
    v : 1D array
        V coordinates [wavelengths]
    eps : float, optional
        NUFFT precision
    """

    # --- Uniform FFT ---
    I0 = np.fft.ifftshift(I) # 画像は中心が(0,0)想定なので、FFT前に ifftshift
    Fk = np.fft.fft2(I0) / I0.size  # 2D FFT (uniform grid)
    Fk = np.fft.fftshift(Fk) # 戻す

    # exp(-2π i (u l + v m))に合わせて補間
    xj = (2.0*np.pi) * u # * np.deg2rad(ps_arcsec/3600) 
    yj = (2.0*np.pi) * v # * np.deg2rad(ps_arcsec/3600) 

    # --- FINUFFT（type-2）---
    V = nufft2d2(xj, yj, Fk, isign=-1, eps=eps)  # V(0,0) = sum I(x,y) になるように正規化

    return V


def image_to_vis_finufft_type3(
    I,                # 2D image: [Jy/pix] 
    xx_as, yy_as,     # 2D grids [arcsec]（位相中心=0）
    kx, ky,           # 1D arrays [wavelengths]
    *,
    eps=1e-6
):
    """
    画像 I(x,y) から 非等間隔(u,v)の視線データ V(u,v) を計算する（NUFFT type-3）
    ****位相中心はxx_as, yy_asに依存するので、fftshiftは不要****
    Parameters
    ----------
    I : 2D array
        Input image [Jy/pix] or [Jy/sr] ** NOT BRIGHTNESS ** : PB correction済みとする
    xx_as, yy_as : 2D arrays
        Image plane coordinates [arcsec] (phase center = 0)
    eps : float, optional
        NUFFT precision
    """

    # xj = (xx_as * ARCSEC2RAD).ravel()  # [rad] --> 外側で1回だけ実行
    # yj = (yy_as * ARCSEC2RAD).ravel()  # [rad]
    cj = I.ravel().astype(np.complex64) 

    # kx = 2.0 * np.pi * u
    # ky = -2.0 * np.pi * v # なぜかここ反転させるとあう
    
    V = nufft2d3(x=xx_as, y=yy_as, c=cj, s=kx, t=ky, isign=+1, eps=eps)
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



# imaging to make a dirty map
def vis_to_image_finufft_type3(
    u, v,           # 1D arrays [wavelengths] uの向きはlと同じで、東向きが正
    V,              # 1D array [Jy]
    V_weight,       # 1D array [weights]
    xx_as, yy_as,   # 2D grids [arcsec]（位相中心=0）
    eps = 1e-6
):
    """
    Non-uniform V(u,v) --> I(x,y) （NUFFT type-3）
    I(l,m) = Re[ Σ w_i V_i exp(+2πi (u_i l + v_i m)) ] / Σ w_i
    ****位相中心はxx_as, yy_asに依存するので、fftshiftは不要****
    CASA MSに入っているuは東向きが正のようだ。従って出力される画像も右向きが正になる。
    Parameters
    ----------
    u, v : 1D array
        U, V coordinates [wavelengths]
    V : 1D array
        Visibilities [Jy]
    V_weight : 1D array
        Weights
    xx_as, yy_as : 2D arrays
        Image plane coordinates [arcsec] (phase center = 0)
    eps : float, optional
        NUFFT precision
    """

    kx = 2.0 * np.pi * u
    ky = 2.0 * np.pi * v
    cj = (V * V_weight).astype(np.complex128)  # [Jy]
    # cj = V.astype(np.complex128)  # [Jy] # uniform weighting

    xj = (xx_as * ARCSEC2RAD).ravel()  # [rad]
    yj = (yy_as * ARCSEC2RAD).ravel()  # [rad]

    I = nufft2d3(x=kx, y=ky, c=cj, s=xj, t=yj, isign=+1, eps=eps).reshape(xx_as.shape)
    I /= np.sum(V_weight) # この規格化はbeamで規格化しているのと同じ

    # beamも作成する
    # ビームの計算はIと同じだが、cjをV_weightにする
    cj_beam = V_weight.astype(np.complex128)
    beam = nufft2d3(x=kx, y=ky, c=cj_beam, s=xj, t=yj, isign=+1, eps=eps).reshape(xx_as.shape)
    beam /= np.sum(V_weight)

    return I.real, beam.real 
    # return I.real[:, ::-1], beam.real # x軸反転
