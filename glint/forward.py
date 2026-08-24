"""
Forward modeling of lensed galaxies in the image plane and in the uv plane.
"""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
import warnings
import numpy as np
from typing import Optional, Tuple
from scipy.signal import fftconvolve
from scipy.fft import fftn, ifftn, rfftn, irfftn, next_fast_len
from .context import ImageContext, UVContext
from . import lensing as ls
from .source import (
    make_rotating_disk_cube,
    Vrot_Courteau1997,
    sersic2d,
)
from .mass_model import (
    r200_kpc_from_m200,
    vcirc2_exponential_disk,
    vcirc2_hernquist_bulge,
    vcirc2_nfw_halo,
    vcirc2_disk_bulge_halo,
)
from .model_parameters import ModelParameters
from .pressure_support import VerticalModel, pressure_supported_rotation_velocity

#--------------------------
# 2D data
#--------------------------

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
    lensed_map_conv  : (ny_img, nx_img)  Jy/beam when ctx.beam has peak=1
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
    pa_rad = np.deg2rad(pa_deg)
    source_map = sersic2d(
        x=ctx.xx_src, y=ctx.yy_src,
        I=F_0,
        x0=x_s, y0=y_s,
        ellip=ellip, pa_rad=pa_rad,
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


#--------------------------
# 3D data
#--------------------------
def convolve_cube_spatial(cube: np.ndarray, psf_kernel: np.ndarray) -> np.ndarray:
    """Convolve a cube spatially with a common or channel-dependent PSF.

    ``psf_kernel`` may be either ``(ny, nx)`` or ``(nchan, ny, nx)``.  No
    convolution is performed along the spectral axis.

    kernelの原点はパッケージのn/2中心規約に従いindex k//2に置く（CASA PSFの
    peak位置と同じ）。奇数kernelでは(k-1)//2と同値でscipyの'same'と一致するが、
    偶数kernel（CASA tcleanのPSFなど）では'same'は1 pixelずれるため使わない。
    """
    kernel = np.asarray(psf_kernel)
    if kernel.ndim == 2:
        kernel = kernel[None, :, :]
    if kernel.ndim != 3 or kernel.shape[0] not in (1, cube.shape[0]):
        raise ValueError("psf_kernel must be 2D or have one plane per cube channel")
    ky, kx = kernel.shape[-2:]
    ny, nx = cube.shape[-2:]
    full = fftconvolve(cube, kernel, mode="full", axes=(-2, -1))
    y0, x0 = ky // 2, kx // 2
    return full[..., y0:y0 + ny, x0:x0 + nx]


def convolve_cube_spatial_cached(cube: np.ndarray, ctx: ImageContext) -> np.ndarray:
    """Spatially convolve a cube using the PSF FFT cached by ``ImageContext``.

    The beam is used exactly as supplied; this function does not normalize it.
    The caller must supply a peak-normalized radio beam to convert an input
    in Jy/pixel to Jy/beam.
    """
    if ctx.beam is None or ctx.beam_rfft is None or ctx.fft_shape is None:
        raise ValueError("ctx must contain a beam")

    cube = np.asarray(cube)
    if cube.ndim != 3 or cube.shape[1:] != ctx.img_shape:
        raise ValueError(
            f"cube shape {cube.shape} is incompatible with image shape {ctx.img_shape}"
        )

    beam = ctx.beam
    if beam.ndim == 2:
        ky, kx = beam.shape
    else:
        if beam.shape[0] != cube.shape[0]:
            raise ValueError("a 3D beam must have one plane per cube channel")
        _, ky, kx = beam.shape

    transformed = rfftn(cube, s=ctx.fft_shape, axes=(-2, -1))
    convolved = irfftn(
        transformed * ctx.beam_rfft, s=ctx.fft_shape, axes=(-2, -1)
    )
    # kernel原点はn/2規約でindex k//2（CASA PSFのpeak位置）。
    # 旧(k-1)//2はscipyの'same'規約で、偶数kernelだとmodelが+1 pixelずれる。
    # 奇数kernelでは両者は同値なのでGaussian PSF（Stage 2）は影響なし。
    y0, x0 = ky // 2, kx // 2
    ny, nx = ctx.img_shape
    return convolved[:, y0:y0 + ny, x0:x0 + nx]


def cube_to_visibilities(
    cube_jy_per_pixel: np.ndarray,
    uv_ctx: UVContext,
    *,
    executor=None,
) -> np.ndarray:
    """Sample an image cube using the plans selected by ``uv_ctx``.

    The input must be flux per image pixel.  The primary beam stored in
    ``uv_ctx`` is applied before each channel is passed to its FINUFFT plan.
    """
    cube = np.asarray(cube_jy_per_pixel, dtype=np.float32)
    if cube.shape != (uv_ctx.nchan, *uv_ctx.image_shape):
        raise ValueError(
            f"cube shape {cube.shape} is incompatible with "
            f"(nchan, ny, nx)=({uv_ctx.nchan}, "
            f"{uv_ctx.image_shape[0]}, {uv_ctx.image_shape[1]})"
        )
    output = np.empty(uv_ctx.Ntot, dtype=np.complex64)

    if uv_ctx.nufft_type == 2:
        def execute_channel(channel):
            plan = uv_ctx.plans[channel]
            if plan is None:
                return channel, None
            # Planのmode軸はimage[y, x]に合わせてある。
            coefficients = np.asarray(
                cube[channel] * uv_ctx.primary_beam(channel),
                dtype=np.complex64,
                order="C",
            )
            return channel, plan.execute(coefficients)
    else:
        def execute_channel(channel):
            plan = uv_ctx.plans[channel]
            if plan is None:
                return channel, None
            coefficients = (
                (cube[channel] * uv_ctx.primary_beam(channel))
                .ravel()
                .astype(np.complex64, order="C")
            )
            return channel, plan.execute(coefficients)

    channels = range(uv_ctx.nchan)
    results = map(execute_channel, channels) if executor is None else executor.map(
        execute_channel, channels
    )
    for channel, values in results:
        if values is not None:
            output[uv_ctx.slices[channel]] = values
    return output


def mass_rotation_curve(
    radius_kpc: np.ndarray,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
    halo_mass_200_msun: float,
    halo_radius_200_kpc: float,
    halo_concentration_200: float,
) -> dict[str, np.ndarray]:
    """Return disk, Hernquist-bulge, NFW-halo, and total speeds [km/s].

    This is the physical replacement for an empirical rotation-curve
    parameterization. Cosmology is kept explicit: callers supply a virial
    radius consistent with ``halo_mass_200_msun`` and their adopted redshift.
    """
    disk_velocity2 = vcirc2_exponential_disk(
        radius_kpc,
        mass_msun=disk_mass_msun,
        scale_radius_kpc=disk_scale_radius_kpc,
    )
    bulge_velocity2 = vcirc2_hernquist_bulge(
        radius_kpc,
        mass_msun=bulge_mass_msun,
        effective_radius_kpc=bulge_effective_radius_kpc,
    )
    halo_velocity2 = vcirc2_nfw_halo(
        radius_kpc,
        mass_200_msun=halo_mass_200_msun,
        radius_200_kpc=halo_radius_200_kpc,
        concentration_200=halo_concentration_200,
    )
    return {
        "disk_kms": np.sqrt(np.maximum(disk_velocity2, 0.0)),
        "bulge_kms": np.sqrt(np.maximum(bulge_velocity2, 0.0)),
        "halo_kms": np.sqrt(np.maximum(halo_velocity2, 0.0)),
        "total_kms": np.sqrt(
            np.maximum(disk_velocity2 + bulge_velocity2 + halo_velocity2, 0.0)
        ),
    }


def forward_model_3D_mass_intrinsic(
    *,
    xx_arcsec: np.ndarray,
    yy_arcsec: np.ndarray,
    radius_arcsec: np.ndarray,
    kpc_per_arcsec: float,
    velocity_channels_kms: np.ndarray,
    surface_brightness_profile: np.ndarray,
    velocity_dispersion_profile_kms: np.ndarray,
    surface_density_scale_radius_arcsec: float | None = None,
    velocity_dispersion_scale_radius_arcsec: float | None = None,
    pressure_support_vertical_model: VerticalModel | None = None,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
    halo_mass_200_msun: float,
    halo_radius_200_kpc: float,
    halo_concentration_200: float,
    inclination_deg: float,
    position_angle_deg: float,
    x_center_arcsec: float = 0.0,
    y_center_arcsec: float = 0.0,
    systemic_velocity_kms: float = 0.0,
    raw_channel_spacing_kms: float = 0.0,
    radius_max_arcsec: float | None = None,
) -> np.ndarray:
    """Generate an unlensed, unconvolved cube from physical mass components.

    Surface brightness is used as the gas-density proxy when pressure support
    is enabled.
    """
    total_velocity2 = vcirc2_disk_bulge_halo(
        radius_kpc=radius_arcsec * kpc_per_arcsec,
        disk_mass_msun=disk_mass_msun,
        disk_scale_radius_kpc=disk_scale_radius_kpc,
        bulge_mass_msun=bulge_mass_msun,
        bulge_effective_radius_kpc=bulge_effective_radius_kpc,
        halo_mass_200_msun=halo_mass_200_msun,
        halo_radius_200_kpc=halo_radius_200_kpc,
        halo_concentration_200=halo_concentration_200,
    )
    total_velocity_kms = np.sqrt(np.maximum(total_velocity2, 0.0))
    if pressure_support_vertical_model is not None:
        total_velocity_kms = pressure_supported_rotation_velocity(
            radius_arcsec,
            total_velocity_kms,
            velocity_dispersion_profile_kms,
            surface_density_scale_radius_arcsec,
            velocity_dispersion_scale_radius_arcsec,
            vertical_model=pressure_support_vertical_model,
        )
    return make_rotating_disk_cube(
        XX=xx_arcsec,
        YY=yy_arcsec,
        x0=x_center_arcsec,
        y0=y_center_arcsec,
        vchan_kms=velocity_channels_kms,
        raw_channel_spacing_kms=raw_channel_spacing_kms,
        inc_deg=inclination_deg,
        pa_deg=position_angle_deg,
        radius=radius_arcsec,
        sb_profile=surface_brightness_profile,
        vrot_profile=total_velocity_kms,
        sigma_profile=velocity_dispersion_profile_kms,
        systemic_kms=systemic_velocity_kms,
        rmax_as=radius_max_arcsec,
    )


'''
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
    lensed_cube_conv  : (nchan, ny_img, nx_img)  Jy/beam when ctx.beam has peak=1
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
    # F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    # F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, \
    F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
        b, q_l, pa_l, log_gamma, pa_gamma = p
        # b, log_gamma, pa_gamma = p


    # deflection angle
    alpha_x_as, alpha_y_as = ls.deflection_SIE_plus_ES(
        xx=ctx.xx_img, yy=ctx.yy_img,
        # x0=x_l, y0=y_l, b=b, q=q_l, pa=pa_l,
        x0=ctx.x0_l, y0=ctx.y0_l, b=b, q=q_l, pa=pa_l, # lens centerはHST Gaussian fitで固定している
        # x0=ctx.x0_l, y0=ctx.y0_l, b=b, q=1, pa=0, # lens centerはHST Gaussian fitで固定している
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
        raw_channel_spacing_kms=ctx.raw_channel_spacing_kms,
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
'''




'''
def _make_lensed_cube_for_vis(params: np.ndarray, img_ctx: ImageContext) -> np.ndarray:
    """Build the primary-beam-free lensed cube in Jy/pixel."""
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
    # F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
    # F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, \
    F_0, pa_deg, r_scale, v_c, r_turn, beta_curve, gamma_curve, sigma_0, r_sigma, vsys_kms, \
        b, q_l, pa_l, log_gamma, pa_gamma = p
        # b, log_gamma, pa_gamma = p


    # deflection angle
    alpha_x_as, alpha_y_as = ls.deflection_SIE_plus_ES(
        xx=img_ctx.xx_img, yy=img_ctx.yy_img,
        # x0=x_l, y0=y_l, b=b, q=q_l, pa=pa_l,
        x0=img_ctx.x0_l, y0=img_ctx.y0_l, b=b, q=q_l, pa=pa_l, # lens centerはHST Gaussian fitで固定している
        # x0=img_ctx.x0_l, y0=img_ctx.y0_l, b=b, q=1, pa=0, # lens centerはHST Gaussian fitで固定している
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
        raw_channel_spacing_kms=img_ctx.raw_channel_spacing_kms,
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
    
    
    # Jy/arcsec^2 -> Jy/pix
    lensed_cube *= (img_ctx.pixsize_img**2)

    return lensed_cube


def forward_model_3D_vis(
    params: np.ndarray,
    img_ctx: ImageContext,
    uv_ctx: UVContext,
) -> np.ndarray:
    """Evaluate the visibility model using the NUFFT type in ``uv_ctx``."""
    cube = _make_lensed_cube_for_vis(params, img_ctx)
    return cube_to_visibilities(cube, uv_ctx)
'''


#--------------------------
# End-to-end forward model
#--------------------------
@dataclass(frozen=True, slots=True)
class ForwardModel:
    """パラメータ→モデル生成の一気通貫API。fitの間ずっと固定な状態を全て保持する。

    幾何・スペクトル軸・モデル仮定を1つに束ね、ModelParametersから
    make_intrinsic_cube → make_lensed_cube → make_convolved_lensed_cube /
    make_visibilities の各段階を直接生成する。
    観測側のuv演算子(UVContext)はstatefulなFINUFFT planを含み
    threadごとに専有する必要があるため、このクラスには持たせず引数で渡す。
    """

    # grids [arcsec]（make_grid_arcsecで作る。float64のまま保持する）
    xx_img: np.ndarray
    yy_img: np.ndarray
    xx_src: np.ndarray
    yy_src: np.ndarray

    # pixel scales [arcsec/pix]
    pixsize_img: float
    pixsize_src: float

    # spectral axis
    vchan_kms: np.ndarray             # (nchan,) 昇順
    raw_channel_spacing_kms: float    # online averaging前のraw channel spacing [km/s]

    # radial grid [arcsec]（profileのlookup table。source pixel以上の細かさにする）
    radius_arcsec: np.ndarray

    # モデル仮定
    kpc_per_arcsec: float
    critical_density_msun_kpc3: float
    halo_concentration_200: float
    vertical_model: Optional[VerticalModel] = None  # Noneならpressure supportなし(v_rot=v_circ)
    lensing_interpolation_order: int = 1

    # source-plane origin [arcsec]
    x0_src: float = 0.0
    y0_src: float = 0.0

    # beam kernel（image-domain fit用。uv fitではNone）
    beam: Optional[np.ndarray] = None
    # cached FFT of beam
    beam_rfft: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    fft_shape: Optional[Tuple[int, int]] = field(default=None, init=False)

    def __post_init__(self):
        for name in ("xx_img", "yy_img", "xx_src", "yy_src"):
            grid = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, grid)
            if grid.ndim != 2 or 0 in grid.shape:
                raise ValueError(f"{name} must be a non-empty 2D array.")
            if not np.all(np.isfinite(grid)):
                raise ValueError(f"{name} contains NaN or inf.")
        if self.xx_img.shape != self.yy_img.shape:
            raise ValueError("xx_img and yy_img must have same shape.")
        if self.xx_src.shape != self.yy_src.shape:
            raise ValueError("xx_src and yy_src must have same shape.")

        for name in ("pixsize_img", "pixsize_src", "kpc_per_arcsec",
                     "critical_density_msun_kpc3", "halo_concentration_200"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0.")

        vchan = np.asarray(self.vchan_kms, dtype=float)
        object.__setattr__(self, "vchan_kms", vchan)
        if vchan.ndim != 1 or vchan.size < 2 or not np.all(np.isfinite(vchan)):
            raise ValueError("vchan_kms must be a finite 1D array with >= 2 channels.")
        if not np.all(np.diff(vchan) > 0):
            # make_rotating_disk_cube は昇順チャネルを前提とする
            raise ValueError("vchan_kms must be strictly increasing.")

        radius = np.asarray(self.radius_arcsec, dtype=float)
        object.__setattr__(self, "radius_arcsec", radius)
        if radius.ndim != 1 or radius.size < 2 or not np.all(np.isfinite(radius)):
            raise ValueError("radius_arcsec must be a finite 1D array.")
        if radius[0] < 0 or not np.all(np.diff(radius) > 0):
            raise ValueError("radius_arcsec must be non-negative and increasing.")

        if not np.isfinite(self.raw_channel_spacing_kms) or self.raw_channel_spacing_kms < 0:
            raise ValueError("raw_channel_spacing_kms must be finite and >= 0.")
        if self.vertical_model is not None and self.vertical_model not in (
            "constant_scale_height", "self_gravitating_hydrostatic",
        ):
            raise ValueError(
                "vertical_model must be None, 'constant_scale_height', "
                "or 'self_gravitating_hydrostatic'."
            )
        if int(self.lensing_interpolation_order) < 0:
            raise ValueError("lensing_interpolation_order must be >= 0.")

        # radius_max が source grid に収まっているかチェック
        margin_x = min(self.x0_src - self.xx_src.min(), self.xx_src.max() - self.x0_src)
        margin_y = min(self.y0_src - self.yy_src.min(), self.yy_src.max() - self.y0_src)
        if radius[-1] > min(margin_x, margin_y):
            warnings.warn(
                f"radius_arcsec max ({radius[-1]:.3f}\") exceeds the source-grid "
                f"half extent ({min(margin_x, margin_y):.3f}\"); the model emission "
                "is truncated by the grid boundary.",
                stacklevel=2,
            )

        if self.beam is not None:
            beam = np.asarray(self.beam, dtype=np.float32, order="C")
            object.__setattr__(self, "beam", beam)
            ny, nx = self.xx_img.shape
            if beam.ndim == 2:
                ky, kx = beam.shape
            elif beam.ndim == 3:
                nbeam, ky, kx = beam.shape
                if nbeam != self.nchan:
                    raise ValueError("a 3D beam must have one plane per velocity channel.")
            else:
                raise ValueError("beam must be 2D or 3D array.")
            if ky == 0 or kx == 0 or not np.all(np.isfinite(beam)):
                raise ValueError("beam must be finite with non-empty spatial dimensions.")
            # padding sizeとcrop規約(s = k//2)の詳細はImageContextの同処理のコメントを参照
            Ly = next_fast_len(max(ny + ky - 1 - ky // 2, ky // 2 + ny))
            Lx = next_fast_len(max(nx + kx - 1 - kx // 2, kx // 2 + nx))
            object.__setattr__(self, "beam_rfft", rfftn(beam, s=(Ly, Lx), axes=(-2, -1)))
            object.__setattr__(self, "fft_shape", (Ly, Lx))

    @property
    def nchan(self) -> int:
        return int(self.vchan_kms.size)

    @property
    def img_shape(self) -> Tuple[int, int]:
        return (int(self.xx_img.shape[0]), int(self.xx_img.shape[1]))

    @property
    def radius_kpc(self) -> np.ndarray:
        return self.radius_arcsec * self.kpc_per_arcsec

    def physical_keywords(self, parameters: ModelParameters) -> dict:
        # Sampling parameterをdisk・bulge・NFW haloの物理量へ変換する。
        baryonic_mass = 10.0 ** parameters.log10_m_baryon
        bulge_mass = parameters.bulge_to_total * baryonic_mass
        halo_mass = 10.0 ** parameters.log10_m200
        return {
            'disk_mass_msun': baryonic_mass - bulge_mass,
            'disk_scale_radius_kpc': parameters.disk_scale_radius_kpc,
            'bulge_mass_msun': bulge_mass,
            'bulge_effective_radius_kpc': parameters.bulge_effective_radius_kpc,
            'halo_mass_200_msun': halo_mass,
            'halo_radius_200_kpc': r200_kpc_from_m200(
                halo_mass, self.critical_density_msun_kpc3,
            ),
            'halo_concentration_200': self.halo_concentration_200,
        }

    def lens_keywords(self, parameters: ModelParameters) -> dict:
        return {
            'x0': parameters.lens_x_arcsec, 'y0': parameters.lens_y_arcsec,
            'b': parameters.einstein_radius_arcsec, 'q': parameters.lens_axis_ratio,
            'pa': np.deg2rad(parameters.lens_position_angle_deg),
            'log_gamma': parameters.log10_external_shear,
            'pa_gamma': np.deg2rad(parameters.external_shear_position_angle_deg),
            'kappa': parameters.external_convergence,
        }

    def make_intrinsic_cube(self, parameters: ModelParameters) -> np.ndarray:
        """Mass modelとtracer profileからsource-plane cubeを作る。

        Returns: (nchan, ny_src, nx_src) surface brightness cube
        """
        surface_brightness = parameters.flux_normalization * np.exp(
            -self.radius_arcsec / parameters.surface_brightness_scale_arcsec,
        )
        velocity_dispersion = parameters.velocity_dispersion_center_kms * np.exp(
            -self.radius_arcsec / parameters.velocity_dispersion_scale_arcsec,
        )
        return forward_model_3D_mass_intrinsic(
            xx_arcsec=self.xx_src, yy_arcsec=self.yy_src,
            radius_arcsec=self.radius_arcsec, kpc_per_arcsec=self.kpc_per_arcsec,
            velocity_channels_kms=self.vchan_kms,
            surface_brightness_profile=surface_brightness,
            velocity_dispersion_profile_kms=velocity_dispersion,
            surface_density_scale_radius_arcsec=parameters.surface_brightness_scale_arcsec,
            velocity_dispersion_scale_radius_arcsec=parameters.velocity_dispersion_scale_arcsec,
            pressure_support_vertical_model=self.vertical_model,
            inclination_deg=parameters.inclination_deg,
            position_angle_deg=parameters.position_angle_deg,
            x_center_arcsec=parameters.source_x_arcsec,
            y_center_arcsec=parameters.source_y_arcsec,
            systemic_velocity_kms=parameters.systemic_velocity_kms,
            raw_channel_spacing_kms=self.raw_channel_spacing_kms,
            radius_max_arcsec=self.radius_arcsec[-1],
            **self.physical_keywords(parameters),
        )

    def make_lensed_cube(self, parameters: ModelParameters) -> np.ndarray:
        """Lens equationでsource cubeをimage planeへ写す。Surface brightnessは保存される。

        Returns: (nchan, ny_img, nx_img) cube [Jy/pixel]
        """
        alpha_x, alpha_y = ls.deflection_SIE_plus_ES(
            self.xx_img, self.yy_img, **self.lens_keywords(parameters),
        )
        image_surface_brightness = ls.map_source_to_image_cube(
            self.xx_img - alpha_x, self.yy_img - alpha_y,
            self.make_intrinsic_cube(parameters), self.pixsize_src,
            order=int(self.lensing_interpolation_order),
            x0_src_arcsec=self.x0_src, y0_src_arcsec=self.y0_src,
        )
        # Jy/arcsec^2 -> Jy/pixel
        return image_surface_brightness * self.pixsize_img**2

    def make_convolved_lensed_cube(
        self, parameters: ModelParameters, scale: float = 1.0,
        primary_beam: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Lensed cubeへ(必要なら)flux scaleとprimary beamを掛け、beamで畳み込む。

        image-domain fit用のend-to-end model（Jy/beam when beam has peak=1）。
        """
        if self.beam is None:
            raise ValueError("make_convolved_lensed_cube requires a beam.")
        cube = self.make_lensed_cube(parameters)
        if scale != 1.0:
            cube = cube * scale
        if primary_beam is not None:
            cube = cube * np.asarray(primary_beam)[None]
        # convolve_cube_spatial_cachedはbeam/beam_rfft/fft_shape/img_shapeを持つ
        # objectを受けるので、このクラス自身をctxとして渡せる（duck typing）
        return convolve_cube_spatial_cached(cube, self)

    def make_visibilities(
        self, parameters: ModelParameters, uv_ctx: UVContext, scale: float = 1.0,
    ) -> np.ndarray:
        """Lensed cubeをuv点でsamplingしたmodel visibilityを返す。

        primary beamはuv_ctxが保持しており、cube_to_visibilities内で適用される。
        """
        cube = self.make_lensed_cube(parameters)
        if scale != 1.0:
            cube = cube * scale
        return cube_to_visibilities(cube, uv_ctx)

    def rotation_curve(self, parameters: ModelParameters) -> dict[str, np.ndarray]:
        # Plot用にdisk・bulge・halo・totalの円運動速度を個別に返す。
        return mass_rotation_curve(
            radius_kpc=self.radius_kpc, **self.physical_keywords(parameters),
        )

    def critical_lines_and_caustics(self, parameters: ModelParameters, min_points: int = 10, min_length: float = 0.0):
        """lens modelのcritical lines（image面）とcaustics（source面）[arcsec]。
        det A=0のcontourを求め、lens equationでsource面へ写す。
        """
        lens_kwargs = self.lens_keywords(parameters)
        _, _, _, _, det_a, _ = ls.analytic_jacobian_lens_mapping(
            self.xx_img, self.yy_img, ls.deflection_jacobian_SIE_plus_ES, lens_kwargs,
        )
        critical_lines = ls._filter_curves(
            ls._extract_zero_contours(self.xx_img, self.yy_img, det_a),
            min_points=min_points, min_length=min_length,
        )
        caustics = ls._filter_curves([
            ls.map_curve_to_source(curve, ls.deflection_SIE_plus_ES, lens_kwargs)
            for curve in critical_lines
        ], min_points=min_points, min_length=min_length)
        return critical_lines, caustics
