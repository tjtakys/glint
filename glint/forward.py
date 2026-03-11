from __future__ import annotations
import numpy as np
from typing import Tuple
from scipy.signal import fftconvolve
from scipy.fft import fftn, ifftn
from .context import ImageContext, UVContext
from . import lensing as ls
from .source import make_rotating_disk_cube, Vrot_Courteau1997, sersic2d



def forward_model_2D_image(params: np.ndarray, ctx: ImageContext) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Image-plane forward model.
    Parameters
    ----------
    params : array-like
      [x_s, y_s, F_0, ellip, pa_deg, r_eff, n,
       (x0, y0), # optional: lens center (HST Gaussian fitとかで固定するなら),
       b, q_l, pa_l, log_gamma, pa_gamma]
      単位: arcsec, deg, km/s, rad

    ctx : ImageContext
      固定の観測/グリッド情報

    Returns
    -------
    source_map       : (ny_src, nx_src)  Jy/arcsec^2
    lensed_map       : (ny_img, nx_img)  Jy/pix
    lensed_map_conv  : (ny_img, nx_img)  (beam_kernelがあれば畳み込み後)
    """
    p = np.asarray(params, dtype=float)

    x_s, y_s, F_0, ellip, pa_deg, r_eff, n, \
        b, q_l, pa_l, log_gamma, pa_gamma = p

    # deflection angle
    alpha_x_as, alpha_y_as = ls.deflection_SIE_plus_ES(
        xx=ctx.xx_img, yy=ctx.yy_img,
        # x0=x_l, y0=y_l, b=b, q=q_l, pa=pa_l,
        x0=ctx.x0_l, y0=ctx.y0_l, b=b, q=q_l, pa=pa_l, # lens centerはHST Gaussian fitで固定している
        log_gamma=log_gamma, pa_gamma=pa_gamma, kappa=0
    )

    beta_x_as, beta_y_as = ctx.xx_img - alpha_x_as, ctx.yy_img - alpha_y_as
    # beta_x_as, beta_y_as = ctx.xx_img, ctx.yy_img # for testing without lensing deflection

    # source model (Jy/arcsec^2)
    source_map = sersic2d(
        x=ctx.xx_src, y=ctx.yy_src,
        I=F_0,
        x0=x_s, y0=y_s,
        ellip=ellip, pa=pa_deg, 
        r_eff=r_eff, 
        n=n
    )

    # map to lensed image (Jy/arcsec^2)
    lensed_map = ls.map_source_to_image(
        beta_x_arcsec=beta_x_as, beta_y_arcsec=beta_y_as,
        source_image=source_map, src_pixscale_arcsec=ctx.pixsize_src, order=1,
        x0_src_arcsec=ctx.x0_src, y0_src_arcsec=ctx.y0_src)
    
    
    # Jy/arcsec^2 -> Jy/pixel
    lensed_map *= (ctx.pixsize_img**2)

    # convolve with clean beam (Jy/pixel -> Jy/beam)
    lensed_map_conv = fftconvolve(lensed_map, ctx.beam, mode='same')
    
    return source_map, lensed_map, lensed_map_conv


def _fftconvolve_nd(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    dataは2D or 3D (cube) のどちらでもOK。3Dの場合、for loopで各チャンネルに2D convolutionを行うより早くなるはず。 --> 試したけど、むしろ遅くなる？？
    """
    if data.ndim == 2:
        data_in = data[None, ...]  # (1, ny, nx)
        squeeze = True
    else:
        data_in = data  # (nchan, ny, nx)
        squeeze = False
    
    nchan, ny, nx = data_in.shape
    ky, kx = kernel.shape
    Ly, Lx = ny + ky - 1, nx + kx - 1

    # FFT
    X = fftn(data_in, s=(Ly, Lx), axes=(-2, -1))
    K = fftn(kernel, s=(Ly, Lx), axes=(-2, -1))
    Y = ifftn(X * K, axes=(-2, -1)).real

    # "same" crop: take the central ny,nx region
    y0 = (ky - 1) // 2
    x0 = (kx - 1) // 2
    out = Y[:, y0:y0 + ny, x0:x0 + nx]

    return out[0] if squeeze else out




def forward_model_3D_image(params: np.ndarray, ctx: ImageContext) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Image-plane forward model.
    Parameters
    ----------
    params : array-like
      [x_s, y_s, F_0, inc_deg, pa_deg, r_scale,
       (x0, y0), # optional: lens center (HST Gaussian fitとかで固定するなら),
       v_c, r_turn, gamma_curve, sigma_0, r_sigma, vsys_kms,
       b, q_l, pa_l, log_gamma, pa_gamma]
      単位: arcsec, deg, km/s, rad

    ctx : ImageContext
      固定の観測/グリッド情報
      beam は 3D cube (nchan, ny_img, nx_img) で与える想定

    Returns
    -------
    source_cube       : (nchan, ny_src, nx_src)  Jy/arcsec^2
    lensed_cube       : (nchan, ny_img, nx_img)  Jy/pix
    lensed_cube_conv  : (nchan, ny_img, nx_img)  (beam_kernelがあれば畳み込み後)
    """
    p = np.asarray(params, dtype=float)

    # x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, gamma_curve, sigma_0, r_sigma, vsys_kms, \
        # x_l, y_l, b, q_l, pa_l, log_gamma, pa_gamma = params
    # x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    # x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    #     b, q_l, pa_l, log_gamma, pa_gamma = p
    # x_s, y_s inc_deg --> fix
    # F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    #     b, q_l, pa_l, log_gamma, pa_gamma = p

    # x_s, y_s inc_deg --> fix
    F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
        b, log_gamma, pa_gamma = p


    # deflection angle
    alpha_x_as, alpha_y_as = ls.deflection_SIE_plus_ES(
        xx=ctx.xx_img, yy=ctx.yy_img,
        # x0=x_l, y0=y_l, b=b, q=q_l, pa=pa_l,
        # x0=ctx.x0_l, y0=ctx.y0_l, b=b, q=q_l, pa=pa_l, # lens centerはHST Gaussian fitで固定している
        x0=ctx.x0_l, y0=ctx.y0_l, b=b, q=1, pa=0, # lens centerはHST Gaussian fitで固定している
        log_gamma=log_gamma, pa_gamma=pa_gamma, kappa=0
    )
    beta_x_as, beta_y_as = ctx.xx_img - alpha_x_as, ctx.yy_img - alpha_y_as

    # source model (Jy/arcsec^2)
    radius = ctx.radius_arcsec
    sb_profile = F_0 * np.exp(-1 * radius/r_scale)
    vrot_profile = Vrot_Courteau1997(r=radius, v_c=v_c, r_turn=r_turn, gamma=gamma_curve, beta=beta_curve)
    # vrot_profile = Vrot_Courteau1997(r=radius, v_c=v_c, r_turn=r_turn, gamma=gamma_curve, beta=1.0)
    sigma_profile = sigma_0 * np.exp(-1 * radius/r_sigma)
    source_cube = make_rotating_disk_cube(
        XX=ctx.xx_src, YY=ctx.yy_src,
        # x0=x_s, y0=y_s,
        x0=ctx.x_s, y0=ctx.y_s,
        vchan_kms=ctx.vchan_kms,
        spec_res_sgm_kms=ctx.spec_res_sgm_kms,
        # inc_deg=inc_deg,
        inc_deg=ctx.inc_deg,
        pa_deg=pa_deg,
        radius=radius,
        sb_profile=sb_profile,
        vrot_profile=vrot_profile,
        sigma_profile=sigma_profile,
        systemic_kms=vsys_kms,
    )

    # map to lensed image (Jy/arcsec^2)
    lensed_cube = ls.map_source_to_image_cube(
        beta_x_arcsec=beta_x_as, beta_y_arcsec=beta_y_as,
        source_cube=source_cube, src_pixscale_arcsec=ctx.pixsize_src, order=2,
        x0_src_arcsec=ctx.x0_src, y0_src_arcsec=ctx.y0_src)
    
    
    # Jy/arcsec^2 -> Jy/pixel
    lensed_cube *= (ctx.pixsize_img**2)

    # convolve with clean beam (Jy/pixel -> Jy/beam)
    # lensed_cube_conv = np.zeros_like(lensed_cube)

    # for i in range(len(ctx.vchan_kms)):
    #     # # image convolution using astropy.convolution.convolve
    #     # lensed_image_conv = convolve2d(lensed_cube[i], beam, mode='same', boundary='fill', fillvalue=0)
    #     # lensed_cube_conv[i] = lensed_image_conv

    #     # fftconvolve (much faster)
    #     lensed_cube_conv[i] = fftconvolve(lensed_cube[i], ctx.beam[i], mode='same')

    Ly, Lx = ctx.fft_shape
    ny, nx = ctx.img_shape
    _, ky, kx = ctx.beam.shape

    # convolve in FFT space
    lensed_cube_fft = fftn(lensed_cube, s=(Ly, Lx), axes=(-2, -1))
    lensed_cube_conv_full = ifftn(lensed_cube_fft * ctx.beam_fft, axes=(-2, -1)).real

    # crop back to "same" shape
    y0 = (ky - 1) // 2
    x0 = (kx - 1) // 2
    lensed_cube_conv = lensed_cube_conv_full[:, y0:y0+ny, x0:x0+nx]

    # 3D convolution (fftconvolve_nd) --- 試したけど、むしろ遅くなる？？
    # lensed_cube_conv = _fftconvolve_nd(lensed_cube, ctx.beam)

    return source_cube, lensed_cube, lensed_cube_conv




def forward_model_3D_vis(params: np.ndarray, img_ctx: ImageContext, uv_ctx: UVContext) -> np.ndarray:
    """
    params = [x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms  # rotating disk 
              x_l, y_l, b, q_l, pa_l, log_gamma, pa_gamma] # lens
    単位は arcsec・rad
    """
    p = np.asarray(params, dtype=float)

    # x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, gamma_curve, sigma_0, r_sigma, vsys_kms, \
        # x_l, y_l, b, q_l, pa_l, log_gamma, pa_gamma = params
    # x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    # x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    #     b, q_l, pa_l, log_gamma, pa_gamma = p
    # x_s, y_s inc_deg --> fix
    # F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    #     b, q_l, pa_l, log_gamma, pa_gamma = p

    # x_s, y_s inc_deg --> fix
    F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
        b, log_gamma, pa_gamma = p


    # deflection angle
    alpha_x_as, alpha_y_as = ls.deflection_SIE_plus_ES(
        xx=img_ctx.xx_img, yy=img_ctx.yy_img,
        # x0=x_l, y0=y_l, b=b, q=q_l, pa=pa_l,
        # x0=img_ctx.x0_l, y0=img_ctx.y0_l, b=b, q=q_l, pa=pa_l, # lens centerはHST Gaussian fitで固定している
        x0=img_ctx.x0_l, y0=img_ctx.y0_l, b=b, q=1, pa=0, # lens centerはHST Gaussian fitで固定している
        log_gamma=log_gamma, pa_gamma=pa_gamma, kappa=0
    )
    beta_x_as, beta_y_as = img_ctx.xx_img - alpha_x_as, img_ctx.yy_img - alpha_y_as

    # source model (Jy/arcsec^2)
    radius = img_ctx.radius_arcsec
    sb_profile = F_0 * np.exp(-1 * radius/r_scale)
    vrot_profile = Vrot_Courteau1997(r=radius, v_c=v_c, r_turn=r_turn, gamma=gamma_curve, beta=beta_curve)
    # vrot_profile = Vrot_Courteau1997(r=radius, v_c=v_c, r_turn=r_turn, gamma=gamma_curve, beta=1.0)
    sigma_profile = sigma_0 * np.exp(-1 * radius/r_sigma)
    source_cube = make_rotating_disk_cube(
        XX=img_ctx.xx_src, YY=img_ctx.yy_src,
        # x0=x_s, y0=y_s,
        x0=img_ctx.x_s, y0=img_ctx.y_s,
        vchan_kms=img_ctx.vchan_kms,
        spec_res_sgm_kms=img_ctx.spec_res_sgm_kms,
        # inc_deg=inc_deg,
        inc_deg=img_ctx.inc_deg,
        pa_deg=pa_deg,
        radius=radius,
        sb_profile=sb_profile,
        vrot_profile=vrot_profile,
        sigma_profile=sigma_profile,
        systemic_kms=vsys_kms,
    )

    # map to lensed image (Jy/arcsec^2)
    lensed_cube = ls.map_source_to_image_cube(
        beta_x_arcsec=beta_x_as, beta_y_arcsec=beta_y_as,
        source_cube=source_cube, src_pixscale_arcsec=img_ctx.pixsize_src, order=2,
        x0_src_arcsec=img_ctx.x0_src, y0_src_arcsec=img_ctx.y0_src)
    
    
    # apply primary beam
    lensed_cube *= uv_ctx.pb

    # Jy/arcsec^2 -> Jy/pix
    lensed_cube *= (img_ctx.pixsize_img**2)

    # to visibility for each channel
    lensed_vis = np.empty(uv_ctx.Ntot, dtype=np.complex64)
    # return lensed_cube

    for i in range(uv_ctx.nchan):
        p = uv_ctx.plans[i]
        if p is None:
            continue

        sl = uv_ctx.slices[i]

        # lensed_vis_i = image_to_vis_finufft_type3(I=lensed_cube[i], xx_as=l_as, yy_as=m_as, u=u[i,m], v=v[i,m], eps=1e-6)
        cj = lensed_cube[i].ravel().astype(np.complex64, order="C")
        lensed_vis[sl] = p.execute(cj)

    return lensed_vis