"""
lensing.py : Calculate lensing deflection angles and related quantities.

Mass models
------
- SIE (Singular Isothermal Ellipsoid): implemented.
- EPL (Elliptical Power-Law): implemented.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# optional JIT  並列化するとおかしくなるのでとりあえず使わずに進める
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False



# -------------------------- Utilities -------------------------- #
def _rot(x: np.ndarray, y: np.ndarray, c:float, s:float) -> tuple[np.ndarray, np.ndarray]:
    """2D rotation of coordinates by cos/sin values (passive)."""
    xr =  c * x + s * y
    yr = -s * x + c * y
    return xr, yr

def _irot(x: np.ndarray, y: np.ndarray, c:float, s:float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse 2D rotation of coordinates by cos/sin values (active)."""
    xi = c * x - s * y
    yi = s * x + c * y
    return xi, yi

# -------------------------- Mass Model -------------------------- #
# SIE deflection (Kormann+94, Keeton+01)
def deflection_SIE(xx, yy, x0, y0, b, q, pa, s=0.0):
    """
    Compute deflection angles for a Singular Isothermal Ellipsoid (SIE) lens model.
    This eq. is based on Keeton+01, which can be found at https://arxiv.org/abs/astro-ph/0102341.
    All lengths in arcsec/pix (common), theta in radians. The b is defined as b/sqrt(q) in
    Keeton's notation.

    Parameters
    ----------
    xx, yy : 2D array
        Image-plane coordinates (same shape).
    x0, y0 : float
        Lens center.
    b : float
        Einstein radius. 
    q : float
        Axis ratio (b/a). Should be in (0, 1).
    pa : float
        Position angle (radians) measured from x-axis to major axis.
    s : float
        The core radius of the lens (defaults to 0.0).

    Returns
    -------
    alpha_x, alpha_y : 2D array
        Deflection angles with same shape as xx, yy.
    """

    # Shift coordinates to lens center
    x_shift = xx - x0
    y_shift = yy - y0

    # Rotate coordinates by position angle (passive)
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_rot, y_rot = _rot(x_shift, y_shift, cos_pa, sin_pa)

    # Calculate deflection components
    q = np.clip(q, 1e-5, 1.0-1e-5)  # Avoid q=0 or 1 exactly
    eps = np.sqrt(1.0 - q**2)
    
    # if eps > 1e-6:
    # General case (SIE)
    psi = np.sqrt(q**2 * (x_rot**2 + s**2) + y_rot**2)
    psi = np.clip(psi, 1e-20, None)  # Avoid division by zero (at origin)
    tx = eps * x_rot / (psi + s)
    ty = eps * y_rot / (psi + q**2 * s)
    # ty = np.clip(ty, -1 + 1e-12, 1 - 1e-12)  # Avoid arctanh(|arg|>=1)　
    alpha_x_rot = b * np.sqrt(q) / eps * np.arctan(tx)
    alpha_y_rot = b * np.sqrt(q) / eps * np.arctanh(ty)
    # else:
    #     # Circular limit (SIS)
    #     psi = np.sqrt(x_rot**2 + y_rot**2)
    #     psi = np.clip(psi, 1e-10, None)  # Avoid division by zero
    #     alpha_x_rot = b * x_rot / psi
    #     alpha_y_rot = b * y_rot / psi
        
    # Rotate deflection back to original frame
    alpha_x, alpha_y = _irot(alpha_x_rot, alpha_y_rot, cos_pa, sin_pa)

    return alpha_x, alpha_y


# External shear
def deflection_ES(xx, yy, x0, y0, gamma1, gamma2, kappa=0.0):
    """
    External shear at the lens center (x0,y0).

    parameters
    ----------
    xx, yy : 2D array
        Image-plane coordinates (same shape).
    x0, y0 : float
        Lens center.
    gamma1, gamma2 : float
        Cartesian shear components.
        gamma1 = gamma cos(2pa_gamma), gamma2 = gamma sin(2pa_gamma) で、pa_gammaは+x（西）から反時計回り。
        ゼロが正規の点になり、log振幅prior + free PAがゼロ近傍に持っていた無限の体積が無くなるので極座標表示よりdynestyのバイアス無くなる
    kappa : float
        Convergence (mass sheet).
    """
    # Shift coordinates to lens center
    x_shift = xx - x0
    y_shift = yy - y0

    # Deflection due to shear and convergence
    alpha_x = x_shift * (kappa + gamma1) + y_shift * gamma2
    alpha_y = x_shift * gamma2 + y_shift * (kappa - gamma1)

    return alpha_x, alpha_y


# Elliptical power-law (EPL; Tessore & Metcalf 2015; https://arxiv.org/abs/1507.01819) deflection.
def _epl_omega(phi, t, q, tol=1e-14, max_terms=5000):
    """
    Tessore & Metcalf 2015, eq. (23, 27, 28, 29)
    汎用的な超幾何関数（scipy.special.hyp2f1）を呼ぶよりも EPLの場合は係数が特殊で漸化式が綺麗に決まるので、4章にある漸化式で評価した方が速い
    項数は f^n < tol となるまで（q=0.88 なら ~12 項）。
    """
    f = (1.0 - q) / (1.0 + q)
    term = np.exp(1j * phi)
    rotate = np.exp(2j * phi)
    omega = term.copy()
    if f <= 0.0:
        return omega  # q=1: 球対称、ω = e^{iφ}
    # 各項の比の絶対値は f 未満なので打ち切り誤差は f^n 以下（t<2; slope<3なら）。q=0.88 で 12 項、q=0.3 で 54 項、q=0.05 で 324 項。
    n_terms = int(np.ceil(np.log(tol) / np.log(f))) + 1
    if n_terms > max_terms:
        import warnings
        warnings.warn(f"_epl_omega: q={q:.4f} needs {n_terms} terms for tol={tol:g}; truncated at {max_terms}.", stacklevel=2)
        n_terms = max_terms
    for n in range(1, n_terms):
        term = term * (-f * (2 * n - (2 - t)) / (2 * n + (2 - t))) *  rotate # Eq. (27, 28, 29)
        omega += term
    return omega


def deflection_EPL(xx, yy, x0, y0, b, q, pa, slope):
    """
    Compute deflection angles for an Elliptical Power-Law (EPL) lens model (Tessore & Metcalf 2015 (eq. 22)).
    slope=2 reproduces ``deflection_SIE`` to machine precision (tests/test_lensing.py).
    All lengths in arcsec/pix (common), theta in radians.

    Parameters
    ----------
    xx, yy : 2D array
        Image-plane coordinates (same shape).
    x0, y0 : float
        Lens center.
    b : float
        Einstein radius (same convention as ``deflection_SIE``; equals the SIS Einstein radius for q=1).
    q : float
        Axis ratio (b/a). Should be in (0, 1]. q=1 gives the spherical power law
        (unlike ``deflection_SIE``, q=1 is allowed).
    pa : float
        Position angle (radians) measured from x-axis (+x = west) to major axis, counter-clockwise.
    slope : float
        3D density slope γ' of the power law (ρ ∝ r^(−γ')). slope=2 is isothermal (SIE);
        slope>2 is steeper (α diverges at the center), slope<2 is shallower (α → 0 at the center).

    Returns
    -------
    alpha_x, alpha_y : 2D array
        Deflection angles with same shape as xx, yy. At the exact lens center (a measure-zero
        singular point) 0 is returned, as in ``deflection_SIE``.
    """
    x_shift = xx - x0
    y_shift = yy - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot, y_rot = _rot(x_shift, y_shift, cos_pa, sin_pa)
    # q=1（球対称）は級数が 1 項で正確に power-law になるので、SIE と違って上側の clip は不要。下側だけ clip
    q = float(np.clip(q, 1e-5, 1.0))
    t = float(slope) - 1.0  # 2D密度の冪、t=1 で等温
    b_tm = b * np.sqrt(q) # so that b follows the same convention as ``deflection_SIE``
    radius = np.hypot(q * x_rot, y_rot) # Eq (3) major-axis frame
    center = radius == 0.0
    safe_radius = np.where(center, 1.0, radius)
    phi = np.arctan2(y_rot, q * x_rot)
    alpha = 2.0 * b_tm / (1.0 + q) * (b_tm / safe_radius) ** (t - 1.0) * _epl_omega(phi, t, q)
    # lens 中心（測度ゼロの特異点）では deflection_SIE と同じく 0 を返す（slope>2 では発散、slope<2 では 0、t=1 では方向依存で不定）。
    alpha = np.where(center, 0.0, alpha)
    alpha_x, alpha_y = _irot(alpha.real, alpha.imag, cos_pa, sin_pa)
    return alpha_x, alpha_y


def deflection_EPL_plus_ES(xx, yy, x0, y0, b, q, pa, gamma1, gamma2, kappa, slope):
    """
    Combined deflection from EPL + external shear.
    pa: in radians, measured from +x (west) to major axis, counter-clockwise.
    """
    alpha_x_epl, alpha_y_epl = deflection_EPL(xx, yy, x0, y0, b, q, pa, slope)
    alpha_x_es, alpha_y_es = deflection_ES(xx, yy, x0, y0, gamma1, gamma2, kappa)
    return alpha_x_epl + alpha_x_es, alpha_y_epl + alpha_y_es


# Combined SIE + external shear
def deflection_SIE_plus_ES(xx, yy, x0, y0, b, q, pa, gamma1, gamma2, kappa):
    """
    Combined deflection from SIE + external shear.
    pa: in radians, measured from +x (west) to major axis, counter-clockwise.
    """
    alpha_x_sie, alpha_y_sie = deflection_SIE(xx, yy, x0, y0, b, q, pa)
    alpha_x_es, alpha_y_es = deflection_ES(xx, yy, x0, y0, gamma1, gamma2, kappa)
    alpha_x = alpha_x_sie + alpha_x_es
    alpha_y = alpha_y_sie + alpha_y_es
    return alpha_x, alpha_y


def deflection_jacobian_SIE(xx, yy, x0, y0, b, q, pa, s=0.0):
    """Analytic derivatives of the SIE deflection field.
    pa: in radians, measured from +x (west) to major axis, counter-clockwise.
    """
    x_shift = xx - x0
    y_shift = yy - y0
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_rot, y_rot = _rot(x_shift, y_shift, cos_pa, sin_pa)

    q = np.clip(q, 1e-5, 1.0 - 1e-5)
    eps = np.sqrt(1.0 - q**2)
    psi = np.sqrt(q**2 * (x_rot**2 + s**2) + y_rot**2)
    singular = psi <= 1e-15
    psi_safe = np.where(singular, 1.0, psi)
    dpsi_dx = q**2 * x_rot / psi_safe
    dpsi_dy = y_rot / psi_safe

    denominator_x = psi_safe + s
    denominator_y = psi_safe + q**2 * s
    tx = eps * x_rot / denominator_x
    ty = eps * y_rot / denominator_y
    scale = b * np.sqrt(q) / eps

    dtx_dx = eps * (denominator_x - x_rot * dpsi_dx) / denominator_x**2
    dtx_dy = -eps * x_rot * dpsi_dy / denominator_x**2
    dty_dx = -eps * y_rot * dpsi_dx / denominator_y**2
    dty_dy = eps * (denominator_y - y_rot * dpsi_dy) / denominator_y**2

    dax_dx_rot = scale * dtx_dx / (1.0 + tx**2)
    dax_dy_rot = scale * dtx_dy / (1.0 + tx**2)
    day_dx_rot = scale * dty_dx / (1.0 - ty**2)
    day_dy_rot = scale * dty_dy / (1.0 - ty**2)

    c = cos_pa
    sn = sin_pa
    dax_dx = c**2 * dax_dx_rot - c * sn * (dax_dy_rot + day_dx_rot) + sn**2 * day_dy_rot
    dax_dy = c * sn * dax_dx_rot + c**2 * dax_dy_rot - sn**2 * day_dx_rot - c * sn * day_dy_rot
    day_dx = c * sn * dax_dx_rot - sn**2 * dax_dy_rot + c**2 * day_dx_rot - c * sn * day_dy_rot
    day_dy = sn**2 * dax_dx_rot + c * sn * (dax_dy_rot + day_dx_rot) + c**2 * day_dy_rot

    return tuple(np.where(singular, np.nan, derivative) for derivative in (
        dax_dx, dax_dy, day_dx, day_dy,
    ))


def deflection_jacobian_ES(xx, yy, gamma1, gamma2, kappa=0.0):
    """Analytic derivatives of the external-shear deflection field.
    gamma1, gamma2: Cartesian shear components.
    """
    shape = np.broadcast_shapes(np.shape(xx), np.shape(yy))
    return (
        np.full(shape, kappa + gamma1),
        np.full(shape, gamma2),
        np.full(shape, gamma2),
        np.full(shape, kappa - gamma1),
    )


def deflection_jacobian_SIE_plus_ES(
        xx, yy, x0, y0, b, q, pa, gamma1, gamma2, kappa):
    """Analytic derivatives of the combined SIE and external-shear deflection.
    pa: in radians, measured from +x (west) to major axis, counter-clockwise.
    """
    sie = deflection_jacobian_SIE(xx, yy, x0, y0, b, q, pa)
    shear = deflection_jacobian_ES(xx, yy, gamma1, gamma2, kappa)
    return tuple(sie_term + shear_term for sie_term, shear_term in zip(sie, shear))


def deflection_jacobian_EPL(xx, yy, x0, y0, b, q, pa, slope):
    """
    critical lines / caustics を計算するときに使う解析式
    pa: in radians, measured from +x (west) to major axis, counter-clockwise.

    Returns
    -------
    A tuple of 2D arrays (dα_x/dx, dα_x/dy, dα_y/dx, dα_y/dy) representing the Jacobian matrix (= 1 - A)
    """
    alpha_x, alpha_y = deflection_EPL(xx, yy, x0, y0, b, q, pa, slope)
    x_shift = xx - x0
    y_shift = yy - y0
    x_rot, y_rot = _rot(x_shift, y_shift, np.cos(pa), np.sin(pa))
    q = float(np.clip(q, 1e-5, 1.0)) 
    t = float(slope) - 1.0
    b_tm = b * np.sqrt(q)
    radius = np.hypot(q * x_rot, y_rot)
    singular = radius == 0.0
    kappa = (2.0 - t) / 2.0 * (b_tm / np.where(singular, 1.0, radius)) ** t # Eq (2)
    z = x_shift + 1j * y_shift
    gamma = ((1.0 - t) * (alpha_x + 1j * alpha_y) - kappa * z) / np.conj(np.where(singular, 1.0, z)) # Eq (16)
    
    # 1 - A matrix
    # lens 中心では deflection_jacobian_SIE と同じく nan
    return tuple(np.where(singular, np.nan, derivative) for derivative in (
        kappa + gamma.real, gamma.imag, gamma.imag, kappa - gamma.real,
    ))


def deflection_jacobian_EPL_plus_ES(xx, yy, x0, y0, b, q, pa, gamma1, gamma2, kappa, slope):
    """Analytic derivatives of the combined EPL and external-shear deflection.
    pa: in radians, measured from +x (west) to major axis, counter-clockwise.
    """
    epl = deflection_jacobian_EPL(xx, yy, x0, y0, b, q, pa, slope)
    shear = deflection_jacobian_ES(xx, yy, gamma1, gamma2, kappa)
    return tuple(epl_term + shear_term for epl_term, shear_term in zip(epl, shear))


# -------------------------- Mapping source to image -------------------------- #
def make_grid_arcsec(nx, ny, pixscale_arcsec, x0_arcsec=0.0, y0_arcsec=0.0):
    """
    Create a 2D coordinate grid in arcseconds.

    Coordinate convention:
      • +x follows increasing column index (image right)
           = west = -RA = -l
      • +y follows increasing row index
           = north = +Dec = +m

    Parameters
    ----------
    nx, ny : int
        Number of pixels in x (columns) and y (rows).
    pixscale_arcsec : float
        Pixel scale in arcsec per pixel.
    x0_arcsec, y0_arcsec : float
        Sky coordinate of the grid center (arcsec offsets).

    Returns
    -------
    xx_as, yy_as : 2D ndarray
        Grids of sky coordinates in arcseconds.
    """
    # Center pixel indices
    # x0_pix = (nx-1)/2.0 # 画像の幾何的中心が原点
    # y0_pix = (ny-1)/2.0
    x0_pix = nx/2.0 # simobserveに合わせるならこっち、以降全てこちらに統一する
    y0_pix = ny/2.0

    yy_idx, xx_idx = np.indices((ny, nx))
    xx_as = (xx_idx - x0_pix) * pixscale_arcsec + x0_arcsec # +x corresponds to West  (-RA)  on the sky
    yy_as = (yy_idx - y0_pix) * pixscale_arcsec + y0_arcsec # 向きが観測画像と合うように

    return xx_as, yy_as


# lens equation (independent of grid shape)
def compute_beta(theta_x_arcsec, theta_y_arcsec, deflector, lens_params):
    """
    Compute source-plane coordinates (beta_x, beta_y) from image-plane coordinates
    (theta_x, theta_y) using the provided deflector function and lens parameters.

    Parameters
    ----------
    theta_x_arcsec, theta_y_arcsec : 2D array
        Image-plane coordinates in arcseconds.
    deflector : function
        Function to compute deflection angles. Should accept (xx, yy, **lens_params).
    lens_params : dict
        Dictionary of lens parameters required by the deflector function.

    Returns
    -------
    beta_x_arcsec, beta_y_arcsec : 2D array
        Source-plane coordinates in arcseconds.
    """

    ax_arcsec, ay_arcsec = deflector(theta_x_arcsec, theta_y_arcsec, **lens_params)
    return theta_x_arcsec - ax_arcsec, theta_y_arcsec - ay_arcsec  # β_x, β_y (arcsec)


# Regular grid
from scipy.ndimage import map_coordinates
def map_source_to_image(beta_x_arcsec, beta_y_arcsec, source_image, 
                        src_pixscale_arcsec, order=1,
                        x0_src_arcsec=0.0, y0_src_arcsec=0.0):
    """
    Map source-plane coordinates to image-plane coordinates using interpolation.

    Parameters
    ----------
    beta_x_arcsec, beta_y_arcsec : 2D array
        Source-plane coordinates in arcseconds.
    source_image : 2D array
        Source-plane image (same shape as beta_x/y).
    src_pixscale_arcsec : float
        Pixel scale in arcseconds/pixel for the source image.
    order : int
        Interpolation order for map_coordinates.
    x0_src_arcsec, y0_src_arcsec : float
        Center coordinates of the source image in arcseconds.

    Returns
    -------
    image_plane : 2D array
        Mapped image-plane image.
    """

    # Source center pixel indices
    ny_src, nx_src = source_image.shape
    # x0_src_pix = (nx_src - 1) / 2.0
    # y0_src_pix = (ny_src - 1) / 2.0
    x0_src_pix = nx_src / 2.0 # make_grid_arcsec に統一する
    y0_src_pix = ny_src / 2.0

    # Convert beta coordinates to pixel indices in the source image
    beta_x_pix = (beta_x_arcsec - x0_src_arcsec) / src_pixscale_arcsec + x0_src_pix
    beta_y_pix = (beta_y_arcsec - y0_src_arcsec) / src_pixscale_arcsec + y0_src_pix # 向きが観測画像と合うように

    # Prepare coordinates for interpolation
    coords = np.array([beta_y_pix.ravel(), beta_x_pix.ravel()])  # (2, N)

    # Interpolate source image at these coordinates
    image_plane_flat = map_coordinates(source_image, coords, order=order, mode='constant', cval=0.0)

    return image_plane_flat.reshape(beta_x_arcsec.shape)


def map_image_to_source(beta_x_arcsec, beta_y_arcsec, image, 
                        src_pixscale_arcsec,
                        x0_src_arcsec=0.0, y0_src_arcsec=0.0, 
                        reducer='average',
                        return_hits=False):
    """
    Map image-plane coordinates to source-plane coordinates using simple average.

    Parameters
    ----------
    beta_x_arcsec, beta_y_arcsec : 2D array
        Source-plane coordinates in arcseconds.
    image : 2D array
        Image-plane image (same shape as beta_x/y).
    src_pixscale_arcsec : float
        Pixel scale in arcseconds/pixel for the source image.
    x0_src_arcsec, y0_src_arcsec : float
        Center coordinates of the source image in arcseconds.
    reducer : str
        Reduction method when multiple image pixels map to the same source pixel.
        'average' (default) or 'sum'.
    return_hits : bool
        If True, also return the hit count map.

    Returns
    -------
    source_plane : 2D array
        Mapped source-plane image.
    hits (optional) : 2D array
        Hit count map.

    """

    # Source center pixel indices
    ny_src, nx_src = image.shape
    # x0_src_pix = (nx_src - 1) / 2.0
    # y0_src_pix = (ny_src - 1) / 2.0
    x0_src_pix = nx_src / 2.0 # make_grid_arcsec に統一する
    y0_src_pix = ny_src / 2.0

    # Convert beta coordinates to pixel indices in the source image
    beta_x_pix = (beta_x_arcsec - x0_src_arcsec) / src_pixscale_arcsec + x0_src_pix
    beta_y_pix = (beta_y_arcsec - y0_src_arcsec) / src_pixscale_arcsec + y0_src_pix

    val = image.ravel().astype(float)
    

    # source pixel
    jx = np.round(beta_x_pix).astype(int).ravel()
    jy = np.round(beta_y_pix).astype(int).ravel()
    
    # 範囲チェック
    mask = (jx >= 0) & (jx < nx_src) & (jy >= 0) & (jy < ny_src) & np.isfinite(val)
    jx = jx[mask]
    jy = jy[mask]
    val = val[mask]

    # allocate
    source = np.zeros((ny_src, nx_src), dtype=float)
    hits = np.zeros((ny_src, nx_src), dtype=np.int32)

    if reducer in ['average', 'sum']:
        np.add.at(source, (jy, jx), val)
        np.add.at(hits, (jy, jx), 1)

        if reducer == 'average':
            # Avoid division by zero
            hits_nonzero = hits > 0
            source[hits_nonzero] /= hits[hits_nonzero]
        else:
            pass  # sum already done
    else:
        raise ValueError(f"Unknown reducer: {reducer}")
    

    if return_hits:
        return source, hits
    return source


def map_source_to_image_cube(
    beta_x_arcsec, beta_y_arcsec, source_cube, 
    src_pixscale_arcsec, order=1,
    x0_src_arcsec=0.0, y0_src_arcsec=0.0):
    """
    Map source-plane coordinates to image-plane coordinates using interpolation.
    3D version: source_cube has shape (nchan, ny, nx)

    Parameters
    ----------
    beta_x_arcsec, beta_y_arcsec : 2D array
        Source-plane coordinates in arcseconds.
    source_cube : 3D array
        Source-plane cube (shape: (nchan, ny, nx)).
    src_pixscale_arcsec : float
        Pixel scale in arcseconds/pixel for the source image.
    order : int
        Interpolation order for map_coordinates.
    x0_src_arcsec, y0_src_arcsec : float
        Center coordinates of the source image in arcseconds.

    Returns
    -------
    image_plane : 3D array
        Mapped image-plane image cube.

    Notes
    -----
    ``mode="constant", cval=0`` で補間する。
    浮動小数点誤差により、本来はsource gridの境界上にある座標が、
    ごくわずかに有効範囲外として扱われる場合がある。
    そのため、source cubeの空間方向の端には十分なゼロ領域があることを前提とする。
    放射がsource gridの端まで達する場合は、補間前に境界付近の座標を明示的に処理すること。
    """

    nch, ny_src, nx_src = source_cube.shape
    ny_img, nx_img = beta_x_arcsec.shape # 各image-plane pixelが対応するsource-plane座標なので_imgで正しい

    # Source center pixel indices
    # x0_src_pix = (nx_src - 1) / 2.0
    # y0_src_pix = (ny_src - 1) / 2.0
    x0_src_pix = nx_src / 2.0 # make_grid_arcsec に統一する
    y0_src_pix = ny_src / 2.0

    # Convert beta coordinates to pixel indices in the source image
    beta_x_pix = (beta_x_arcsec - x0_src_arcsec) / src_pixscale_arcsec + x0_src_pix
    beta_y_pix = (beta_y_arcsec - y0_src_arcsec) / src_pixscale_arcsec + y0_src_pix

    """ 効率悪いので置き換え
    # Interpolate source image at these coordinates
    z_coords = np.arange(nch)[:, None, None]  # (nch, 1, 1)
    z_coords = np.broadcast_to(z_coords,   shape=(nch, ny_img, nx_img))  # (nch, ny_img, nx_img)
    y_coords = np.broadcast_to(beta_y_pix, shape=(nch, ny_img, nx_img))  # (nch, ny_img, nx_img)
    x_coords = np.broadcast_to(beta_x_pix, shape=(nch, ny_img, nx_img))  # (nch, ny_img, nx_img)

    coords = np.stack([z_coords, y_coords, x_coords], axis=0)  # (3, nch, ny_img, nx_img)
    
    # Interpolation (XY方向のみ、Z方向は整数値を指定するので補間されない)
    # map_coordinates は 入力shapeの[1:]のshapeを返す
    image_plane_flat = map_coordinates(source_cube, coords, order=order, mode='constant', cval=0.0, prefilter=(order > 1))

    return image_plane_flat
    """

    if order == 1:
        # β座標は全channel共通なので、bilinearのindex・重み計算を1回だけ行い、4回のfancy-index gatherで全channelへ一括適用する（sli.pyのbuild_matrix_lensingと同じ）
        # 4点stencilがsource gridに完全に収まらないpixelを0にする。map_coordinatesは境界の外側だけをcval=0として部分的に補間するが、source cubeの端はゼロ余白前提（上のNotes参照）なので実質同じ。
        i0 = np.floor(beta_x_pix).astype(np.int64)  # 左下pixelのx index
        j0 = np.floor(beta_y_pix).astype(np.int64)  # 左下pixelのy index
        inside = (
            (i0 >= 0) & (i0 + 1 < nx_src)
            & (j0 >= 0) & (j0 + 1 < ny_src)
        )
        # 範囲外はclipして安全なindexにし、最後にinsideで0にする
        i0_safe = np.clip(i0, 0, nx_src - 2)
        j0_safe = np.clip(j0, 0, ny_src - 2)
        weight_dtype = (
            source_cube.dtype
            if np.issubdtype(source_cube.dtype, np.floating)
            else np.float64
        )
        dx = (beta_x_pix - i0).astype(weight_dtype)
        dy = (beta_y_pix - j0).astype(weight_dtype)
        w00 = ((1.0 - dx) * (1.0 - dy)).ravel()
        w10 = (dx * (1.0 - dy)).ravel()
        w01 = ((1.0 - dx) * dy).ravel()
        w11 = (dx * dy).ravel()
        flat = source_cube.reshape(nch, -1)
        index00 = (j0_safe * nx_src + i0_safe).ravel()  # flatten規約: j*nx + i
        image_flat = (
            flat[:, index00] * w00
            + flat[:, index00 + 1] * w10
            + flat[:, index00 + nx_src] * w01
            + flat[:, index00 + nx_src + 1] * w11
        )
        image_flat *= inside.ravel()
        return image_flat.reshape(nch, ny_img, nx_img)

    # order >= 2 (spline) は従来どおりchannelごとのmap_coordinatesを使う。
    # 全channelで共通のXY座標
    coords = np.stack([beta_y_pix, beta_x_pix])

    image_cube = np.empty((nch, ny_img, nx_img), dtype=source_cube.dtype)

    for channel in range(nch):
        image_cube[channel] = map_coordinates(
            source_cube[channel],
            coords,
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=(order > 1),
        )

    return image_cube

# -------------------------- Calculate critical line and caustics -------------------------- #
"""
det A(theta) = 0 となる theta がcritical lines、causticsはそれをsource planeに戻せばいい。
A = d beta / d theta = I - d alpha / d theta だから、deflectorの微分 np.gradient() を計算すればdet Aを求められる。
しかし、この場合、特異点付近ではdet Aが急激に変化するので、gridの解像度が低いとdet A=0のcontourが正しく求められず、おかしなlineが出現する。
今回はSIEを想定しているので、analyticなdeflectorの微分を直接使う。将来的にもっと複雑なモデルを追加する場合はJaxなどの自動微分を使うのが良いかもしれない。
"""

def jacobian_lens_mapping(xx, yy, alpha_x, alpha_y):
    """
    Compute the Jacobian matrix A = d beta / d theta
    from deflection fields alpha_x, alpha_y on a regular grid.

    Parameters
    ----------
    xx, yy : 2D ndarray
        Regular image-plane coordinate grids [arcsec].
    alpha_x, alpha_y : 2D ndarray
        Deflection fields [arcsec].

    Returns
    -------
    A11, A12, A21, A22 : 2D ndarray
        Components of Jacobian matrix A.
    detA : 2D ndarray
        Determinant of A.
    mu : 2D ndarray
        Signed magnification = 1 / detA
    """
    # grid spacing [arcsec/pix]
    dx = float(np.mean(np.diff(xx[0, :])))
    dy = float(np.mean(np.diff(yy[:, 0])))

    # derivatives of alpha
    # np.gradient returns derivatives along axis 0 (y), axis 1 (x)
    dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, dy, dx)
    dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, dy, dx)

    # A = d beta / d theta = I - d alpha / d theta
    A11 = 1.0 - dalpha_x_dx
    A12 =     - dalpha_x_dy
    A21 =     - dalpha_y_dx
    A22 = 1.0 - dalpha_y_dy

    detA = A11 * A22 - A12 * A21

    mu = np.full_like(detA, np.inf, dtype=float)
    mask = np.abs(detA) > 1e-12
    mu[mask] = 1.0 / detA[mask]

    return A11, A12, A21, A22, detA, mu


def analytic_jacobian_lens_mapping(xx, yy, deflector_jacobian, lens_params):
    """Compute A = d beta / d theta from analytic deflection derivatives."""
    dalpha_x_dx, dalpha_x_dy, dalpha_y_dx, dalpha_y_dy = deflector_jacobian(
        xx, yy, **lens_params,
    )
    A11 = 1.0 - dalpha_x_dx
    A12 = -dalpha_x_dy
    A21 = -dalpha_y_dx
    A22 = 1.0 - dalpha_y_dy
    detA = A11 * A22 - A12 * A21
    mu = np.full_like(detA, np.inf, dtype=float)
    mask = np.isfinite(detA) & (np.abs(detA) > 1e-12)
    mu[mask] = 1.0 / detA[mask]
    mu[~np.isfinite(detA)] = np.nan
    return A11, A12, A21, A22, detA, mu


def _extract_zero_contours(xx, yy, field, level=0.0):
    """
    Extract contour line vertices for field(xx,yy)=level.

    Parameters
    ----------
    xx, yy : 2D ndarray
        Regular coordinate grids.
    field : 2D ndarray
        Scalar field sampled on the same grid.
    level : float
        Contour level.

    Returns
    -------
    segments : list of ndarray
        Each element has shape (Npt, 2), columns = [x, y]
    """
    import contourpy as cpy
    import numpy as np

    # regular grid を仮定
    x1d = xx[0, :]
    y1d = yy[:, 0]

    cg = cpy.contour_generator(
        x=x1d,
        y=y1d,
        z=field,
    )

    lines = cg.lines(level)

    segments = []
    for line in lines:
        if len(line) >= 2:
            segments.append(np.asarray(line, dtype=float))

    return segments


def map_curve_to_source(curve_xy, deflector, lens_params):
    """
    Map a curve in the image plane to the source plane using the lens equation.

    Parameters
    ----------
    curve_xy : ndarray, shape (N,2)
        Curve points in image plane [arcsec], columns = [theta_x, theta_y]
    deflector : callable
        Deflection function, e.g. deflection_SIE_plus_ES
    lens_params : dict
        Parameters passed to deflector

    Returns
    -------
    curve_beta : ndarray, shape (N,2)
        Mapped curve in source plane [arcsec], columns = [beta_x, beta_y]
    """
    tx = curve_xy[:, 0]
    ty = curve_xy[:, 1]

    ax, ay = deflector(tx, ty, **lens_params)
    bx = tx - ax
    by = ty - ay

    return np.column_stack([bx, by])


def _curve_length(curve_xy):
    """
    Compute total polyline length.
    """
    if len(curve_xy) < 2:
        return 0.0
    d = np.diff(curve_xy, axis=0)
    return np.sum(np.hypot(d[:, 0], d[:, 1]))


def _filter_curves(curves, min_points=10, min_length=0.0):
    """
    Filter short/noisy contour segments.
    """
    out = []
    for seg in curves:
        if len(seg) < min_points:
            continue
        if _curve_length(seg) < min_length:
            continue
        out.append(seg)
    return out


def _arcsec_curve_to_pixel(curve_xy, pixscale_arcsec, nx, ny, x0_arcsec=0.0, y0_arcsec=0.0):
    """
    Convert a curve from arcsec coordinates to pixel coordinates.

    Parameters
    ----------
    curve_xy : ndarray, shape (N,2)
        Curve points in arcsec, columns = [x_arcsec, y_arcsec]
    pixscale_arcsec : float
        Pixel scale [arcsec/pix]
    nx, ny : int
        Image size
    x0_arcsec, y0_arcsec : float
        Coordinate of the grid center [arcsec]

    Returns
    -------
    curve_pix : ndarray, shape (N,2)
        Curve points in pixel coordinates, columns = [x_pix, y_pix]
    """
    # x0_pix = (nx - 1) / 2.0
    # y0_pix = (ny - 1) / 2.0
    x0_pix = nx / 2.0 # make_grid_arcsec に統一する
    y0_pix = ny / 2.0

    x_pix = (curve_xy[:, 0] - x0_arcsec) / pixscale_arcsec + x0_pix
    y_pix = (curve_xy[:, 1] - y0_arcsec) / pixscale_arcsec + y0_pix

    return np.column_stack([x_pix, y_pix])


def compute_critical_lines_and_caustics(
        ctx, deflector, lens_params, min_points=10, min_length=0.0,
        deflector_jacobian=None):
    """
    Compute critical lines and caustics using ImageContext.

    Parameters
    ----------
    ctx : ImageContext
        Context containing image-plane and source-plane grids / pixel scales.
    deflector : callable
        Deflection function, e.g. deflection_SIE or deflection_SIE_plus_ES
    lens_params : dict
        Parameters passed to deflector
    deflector_jacobian : callable, optional
        Analytic derivatives of the deflection field. If omitted, use grid
        finite differences for backward compatibility.
    min_points : int
        Minimum number of contour points to keep.
    min_length : float
        Minimum contour length [arcsec] to keep.

    Returns
    -------
    result : dict
        {
          "alpha_x": 2D ndarray,
          "alpha_y": 2D ndarray,
          "detA": 2D ndarray,
          "mu": 2D ndarray,
          "critical_lines": list of ndarray[(N,2)],      # arcsec
          "caustics": list of ndarray[(N,2)],            # arcsec
          "critical_lines_pix": list of ndarray[(N,2)],  # pixel on image plane
          "caustics_pix": list of ndarray[(N,2)],        # pixel on source plane
        }
    """
    # deflection on image plane
    alpha_x, alpha_y = deflector(ctx.xx_img, ctx.yy_img, **lens_params)

    # Jacobian / magnification
    if deflector_jacobian is None:
        _, _, _, _, detA, mu = jacobian_lens_mapping(
            ctx.xx_img, ctx.yy_img, alpha_x, alpha_y,
        )
    else:
        _, _, _, _, detA, mu = analytic_jacobian_lens_mapping(
            ctx.xx_img, ctx.yy_img, deflector_jacobian, lens_params,
        )

    # critical lines in image plane (arcsec)
    critical_lines = _extract_zero_contours(ctx.xx_img, ctx.yy_img, detA, level=0.0)
    critical_lines = _filter_curves(critical_lines, min_points=min_points, min_length=min_length)

    # caustics in source plane (arcsec)
    caustics = [
        map_curve_to_source(curve, deflector, lens_params)
        for curve in critical_lines
    ]
    caustics = _filter_curves(caustics, min_points=min_points, min_length=min_length)

    # convert to pixel coordinates
    ny_img, nx_img = ctx.img_shape
    ny_src, nx_src = ctx.src_shape

    critical_lines_pix = [
        _arcsec_curve_to_pixel(
            seg,
            pixscale_arcsec=ctx.pixsize_img,
            nx=nx_img,
            ny=ny_img,
            x0_arcsec=0.0,
            y0_arcsec=0.0,
        )
        for seg in critical_lines
    ]

    caustics_pix = [
        _arcsec_curve_to_pixel(
            seg,
            pixscale_arcsec=ctx.pixsize_src,
            nx=nx_src,
            ny=ny_src,
            x0_arcsec=ctx.x0_src,
            y0_arcsec=ctx.y0_src,
        )
        for seg in caustics
    ]

    return {
        "alpha_x": alpha_x,
        "alpha_y": alpha_y,
        "detA": detA,
        "mu": mu,
        "critical_lines": critical_lines,
        "caustics": caustics,
        "critical_lines_pix": critical_lines_pix,
        "caustics_pix": caustics_pix,
    }


def critical_lines_and_caustics_EPL_plus_ES(x0, y0, b, q, pa, gamma1, gamma2, kappa, slope, n_phi=2000):
    """
    Critical lines and caustics of EPL + external shear without a grid (contour 不要).

    楕円角 φ（deflection_EPL の phi）を固定すると、u = (b_TM/R)^t として EPL の κ = K u、γ = G(φ) u
    （K = (2−t)/2、G は deflection_jacobian_EPL の式に z = R w(φ)、w = cos φ/q + i sin φ を入れたもの）で、
    外部 shear と kappa は定数。よって det A = (1 − kappa − K u)² − |γ_ext + G u|² = 0 は u の 2 次方程式
        a u² − 2B u + c = 0,  a = K² − |G|²,  B = (1 − kappa) K + Re(γ_ext* G),  c = (1 − kappa)² − |γ_ext|²
    になり、各 φ で解ける。小さい方の正根（R が大きい方）が tangential、a > 0 の方向だけにある大きい方の根が radial。
    SIE（slope=2）は a = 0 で radial は中心に潰れる。slope>2 は a < 0 で tangential のみ。
    |γ_ext| < 1 − kappa（c > 0）を仮定する。

    Parameters
    ----------
    x0, y0, b, q, pa, gamma1, gamma2, kappa, slope :
        Same as ``deflection_EPL_plus_ES``.
    n_phi : int
        Number of points along each curve.

    Returns
    -------
    critical_lines, caustics : list of ndarray[(n_phi, 2)]
        [tangential] or [tangential, radial], in arcsec (columns = [x, y]).
        radial が一部の φ にしか無い場合、無い φ は nan（matplotlib では途切れて描かれる）。
    """
    q = float(np.clip(q, 1e-5, 1.0))  # deflection_EPL と同じ
    t = float(slope) - 1.0
    b_tm = b * np.sqrt(q)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    w = np.cos(phi) / q + 1j * np.sin(phi)  # lens 主軸系で z = R w(φ)
    K = (2.0 - t) / 2.0
    G = (2.0 * (1.0 - t) / (1.0 + q) * _epl_omega(phi, t, q) - K * w) / np.conj(w)
    gamma_ext = (gamma1 + 1j * gamma2) * np.exp(-2j * pa)  # 外部 shear を lens 主軸系へ（spin 2）
    a = K**2 - np.abs(G)**2
    B = (1.0 - kappa) * K + np.real(np.conj(gamma_ext) * G)
    c = (1.0 - kappa)**2 - np.abs(gamma_ext)**2
    discriminant = B**2 - a * c
    root = np.sqrt(np.where(discriminant >= 0.0, discriminant, np.nan))
    u_tangential = c / (B + root)  # = (B − root)/a。この形なら a → 0（SIE）でも割り算が安定
    has_radial = a > 1e-12  # SIE は a = 0 だが丸め誤差で ±1e-17 程度になり、中心に偽の radial が出るので閾値を置く
    u_radial = np.where(has_radial, (B + root) / np.where(has_radial, a, 1.0), np.nan)

    deflector_kwargs = dict(x0=x0, y0=y0, b=b, q=q, pa=pa, gamma1=gamma1, gamma2=gamma2, kappa=kappa, slope=slope)
    critical_lines = []
    caustics = []
    for u in (u_tangential, u_radial):
        u = np.where(u > 0.0, u, np.nan)
        if not np.any(np.isfinite(u)):
            continue
        z_rot = b_tm * u ** (-1.0 / t) * w  # R = b_TM u^(−1/t)
        x, y = _irot(z_rot.real, z_rot.imag, np.cos(pa), np.sin(pa))
        curve = np.column_stack([x + x0, y + y0])
        critical_lines.append(curve)
        caustics.append(map_curve_to_source(curve, deflection_EPL_plus_ES, deflector_kwargs))
    return critical_lines, caustics
