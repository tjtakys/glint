from __future__ import annotations
import numpy as np
from typing import Tuple
from scipy.signal import fftconvolve
from .context import ImageContext
from . import lensing as ls
from .source import make_rotating_disk_cube, Vrot_Courteau1997

def forward_model_image(params: np.ndarray, ctx: ImageContext) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    source_cube       : (nchan, ny_src, nx_src)  Jy/arcsec^2
    lensed_cube       : (nchan, ny_img, nx_img)  Jy/pix
    lensed_cube_conv  : (nchan, ny_img, nx_img)  (beam_kernelがあれば畳み込み後)
    """
    p = np.asarray(params, dtype=float)

    # x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    # x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, gamma_curve, sigma_0, r_sigma, vsys_kms, \
        # x_l, y_l, b, q_l, pa_l, log_gamma, pa_gamma = params
    x_s, y_s, F_0, inc_deg, pa_deg, r_scale, v_c, r_turn, gamma_curve, sigma_0, r_sigma, vsys_kms, \
        b, q_l, pa_l, log_gamma, pa_gamma = p

    # deflection angle
    alpha_x_as, alpha_y_as = ls.deflection_SIE_plus_ES(
        xx=ctx.xx_img, yy=ctx.yy_img,
        # x0=x_l, y0=y_l, b=b, q=q_l, pa=pa_l,
        x0=ctx.x0_l, y0=ctx.y0_l, b=b, q=q_l, pa=pa_l, # lens centerはHST Gaussian fitで固定している
        log_gamma=log_gamma, pa_gamma=pa_gamma, kappa=0
    )
    beta_x_as, beta_y_as = ctx.xx_img - alpha_x_as, ctx.yy_img - alpha_y_as

    # source model (Jy/arcsec^2)
    radius = ctx.radius_arcsec
    sb_profile = F_0 * np.exp(-1 * radius/r_scale)
    # vrot_profile = Vrot_Courteau1997(r=radius, v_c=v_c, r_turn=r_turn, gamma=gamma_curve, beta=beta_curve)
    vrot_profile = Vrot_Courteau1997(r=radius, v_c=v_c, r_turn=r_turn, gamma=gamma_curve, beta=1.0)
    sigma_profile = sigma_0 * np.exp(-1 * radius/r_sigma)
    source_cube = make_rotating_disk_cube(
        XX=ctx.xx_src, YY=ctx.yy_src,
        x0=x_s, y0=y_s,
        vchan_kms=ctx.vchan_kms,
        spec_res_sgm_kms=ctx.spec_res_sgm_kms,
        inc_deg=inc_deg,
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
        source_cube=source_cube, src_pixscale_arcsec=ctx.pixsize_src, order=1,
        x0_src_arcsec=ctx.x0_src, y0_src_arcsec=ctx.y0_src)
    
    
    # Jy/arcsec^2 -> Jy/pixel
    lensed_cube *= (ctx.pixsize_img**2)

    # convolve with clean beam (Jy/pixel -> Jy/beam)
    lensed_cube_conv = np.zeros_like(lensed_cube)
    for i in range(len(ctx.vchan_kms)):
        # image convolution using astropy.convolution.convolve
        # lensed_image_conv = convolve2d(lensed_cube[i], beam, mode='same', boundary='fill', fillvalue=0)
        # lensed_cube_conv[i] = lensed_image_conv

        # fftconvolve (much faster)
        lensed_cube_conv[i] = fftconvolve(lensed_cube[i], ctx.beam, mode='same')

    return source_cube, lensed_cube, lensed_cube_conv


