"""
lensing.py : Calculate lensing deflection angles and related quantities.

Models
------
- SIE (Singular Isothermal Ellipsoid): implemented.
"""

from __future__ import annotations
import numpy as np

# optional JIT
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
def deflection_ES(xx, yy, x0, y0, log_gamma, pa_gamma, kappa=0.0):
    """
    External shear at the lens center (x0,y0).

    parameters
    ----------
    xx, yy : 2D array
        Image-plane coordinates (same shape).
    x0, y0 : float
        Lens center.
    log_gamma : float
        Shear amplitude in log10 scale.
        gamma = 10**log_gamma
    pa_gamma : float
        Shear position angle.  gamma1 = gamma cos(2pa_gamma), gamma2 = gamma sin(2pa_gamma).
        pa_gamma ∈ [0, π/2) is sufficient
    kappa : float
        Convergence (mass sheet).
    """
    # Shift coordinates to lens center
    x_shift = xx - x0
    y_shift = yy - y0

    # Deflection due to shear and convergence
    gamma1 = 10**log_gamma * np.cos(2.0 * pa_gamma)
    gamma2 = 10**log_gamma * np.sin(2.0 * pa_gamma)

    alpha_x = x_shift * (kappa + gamma1) + y_shift * gamma2
    alpha_y = x_shift * gamma2 + y_shift * (kappa - gamma1)

    return alpha_x, alpha_y


# Combined SIE + external shear
def deflection_SIE_plus_ES(xx, yy, x0, y0, b, q, pa, log_gamma, pa_gamma, kappa):
    """
    Combined deflection from SIE + external shear.
    """
    alpha_x_sie, alpha_y_sie = deflection_SIE(xx, yy, x0, y0, b, q, pa)
    alpha_x_es, alpha_y_es = deflection_ES(xx, yy, x0, y0, log_gamma, pa_gamma, kappa)
    alpha_x = alpha_x_sie + alpha_x_es
    alpha_y = alpha_y_sie + alpha_y_es
    return alpha_x, alpha_y

    

# -------------------------- Mapping source to image -------------------------- #
def make_grid_arcsec(nx, ny, pixscale_arcsec, x0_arcsec=0.0, y0_arcsec=0.0):
    """
    Create a 2D coordinate grid in arcseconds with a sky-like convention.

    Coordinate convention
    ---------------------
    • +x direction = to the RIGHT in the array = WEST on the sky (= -RA)
    • +y direction = UPWARD on the sky = +Dec (North)

    Important:
    NumPy arrays index rows downward (row index increases toward the bottom).
    To make +y correspond to North (up on the sky), we introduce a minus sign
    in the y-coordinate definition below.

    This ensures that:
        increasing x  → move right  → RA decreases
        increasing y  → move up     → Dec increases

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
    x0_pix = (nx-1)/2.0
    y0_pix = (ny-1)/2.0

    yy_idx, xx_idx = np.indices((ny, nx))
    xx_as =  (xx_idx - x0_pix) * pixscale_arcsec + x0_arcsec # +x corresponds to West  (-RA)  on the sky
    yy_as = -(yy_idx - y0_pix) * pixscale_arcsec + y0_arcsec # +y corresponds to North (+Dec) on the sky

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
    x0_src_pix = (nx_src - 1) / 2.0
    y0_src_pix = (ny_src - 1) / 2.0

    # Convert beta coordinates to pixel indices in the source image
    beta_x_pix = (beta_x_arcsec - x0_src_arcsec) / src_pixscale_arcsec + x0_src_pix
    beta_y_pix = (beta_y_arcsec - y0_src_arcsec) / src_pixscale_arcsec + y0_src_pix

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
    x0_src_pix = (nx_src - 1) / 2.0
    y0_src_pix = (ny_src - 1) / 2.0

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
    """

    nch, ny_src, nx_src = source_cube.shape
    ny_img, nx_img = beta_x_arcsec.shape

    # Source center pixel indices
    x0_src_pix = (nx_src - 1) / 2.0
    y0_src_pix = (ny_src - 1) / 2.0

    # Convert beta coordinates to pixel indices in the source image
    beta_x_pix = (beta_x_arcsec - x0_src_arcsec) / src_pixscale_arcsec + x0_src_pix
    beta_y_pix = (beta_y_arcsec - y0_src_arcsec) / src_pixscale_arcsec + y0_src_pix


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

# -------------------------- Calculate critical line and caustics -------------------------- #
# from scipy.ndimage import gaussian_filter
# try:
#     # 等高線の抽出（無ければ matplotlib の Contour を後述の代替で）
#     from skimage.measure import find_contours
#     _HAS_SKIMAGE = True
# except Exception:
#     _HAS_SKIMAGE = False

# # 既存: def deflection_SIE_plus_ES(xx, yy, x0, y0, b, q, pa, gamma, theta_gamma, kappa=0): ...

# def _jacobian_from_alpha(alpha_x, alpha_y, pixscale_arcsec):
#     """
#     2D場 α(θ) からヤコビアン ∂α/∂θ を数値微分で作る。
#     alpha_x, alpha_y: shape (ny, nx)
#     pixscale_arcsec: ピクセル間隔[arcsec]（等方ピクセルを想定）
#     戻り値: dax_dx, dax_dy, day_dx, day_dy
#     """
#     dy = dx = float(pixscale_arcsec)
#     # np.gradient の第1引数は (axis=0の間隔, axis=1の間隔) = (dy, dx)
#     dax_dy, dax_dx = np.gradient(alpha_x, dy, dx)
#     day_dy, day_dx = np.gradient(alpha_y, dy, dx)
#     return dax_dx, dax_dy, day_dx, day_dy

# def kappa_gamma_from_model(xx, yy, pixscale_arcsec, *,
#                            x0, y0, b, q, pa, gamma, theta_gamma, kappa0=0.0,
#                            smooth_sigma_pix=None):
#     """
#     与えたレンズ（SIE+ES）の κ, γ1, γ2 と detA を返すユーティリティ。
#     """
#     ax, ay = deflection_SIE_plus_ES(xx=xx, yy=yy, x0=x0, y0=y0, b=b, q=q, pa=pa,
#                                     gamma=gamma, theta_gamma=theta_gamma, kappa=kappa0)
#     if smooth_sigma_pix and smooth_sigma_pix > 0:
#         ax = gaussian_filter(ax, smooth_sigma_pix)
#         ay = gaussian_filter(ay, smooth_sigma_pix)

#     dax_dx, dax_dy, day_dx, day_dy = _jacobian_from_alpha(ax, ay, pixscale_arcsec)

#     # 収束とシア（標準の定義）
#     kappa = 0.5 * (dax_dx + day_dy)
#     gamma1 = 0.5 * (dax_dx - day_dy)
#     gamma2 = 0.5 * (dax_dy + day_dx)

#     detA = (1.0 - kappa)**2 - (gamma1**2 + gamma2**2)
#     return kappa, gamma1, gamma2, detA

# def critical_curve_mask(xx, yy, pixscale_arcsec, *,
#                         x0, y0, b, q, pa, gamma, theta_gamma, kappa0=0.0,
#                         thresh=0.0, eps_rel=5e-3, smooth_sigma_pix=0.0):
#     """
#     detA ≃ 0 の画素を True とするブールマスクを返す。
#     - thresh=0 を推奨、代わりに eps_rel × |detA|max で相対閾値を使う。
#     """
#     _, _, _, detA = kappa_gamma_from_model(
#         xx, yy, pixscale_arcsec,
#         x0=x0, y0=y0, b=b, q=q, pa=pa, gamma=gamma, theta_gamma=theta_gamma,
#         kappa0=kappa0, smooth_sigma_pix=smooth_sigma_pix
#     )
#     detA = np.asarray(detA)
#     # 相対しきい値
#     tol = max(abs(detA).max() * eps_rel, 1e-12)
#     mask = np.abs(detA - thresh) <= tol
#     return detA, mask

# def critical_curves(xx, yy, pixscale_arcsec, *,
#                     x0, y0, b, q, pa, gamma, theta_gamma, kappa0=0.0,
#                     smooth_sigma_pix=1.0):
#     """
#     連続曲線としてのクリティカルライン座標を返す。
#     戻り値: list of arrays [ (N_i, 2) ] で各成分が (y, x) ピクセル座標。
#     """
#     _, _, _, detA = kappa_gamma_from_model(
#         xx, yy, pixscale_arcsec,
#         x0=x0, y0=y0, b=b, q=q, pa=pa, gamma=gamma, theta_gamma=theta_gamma,
#         kappa0=kappa0, smooth_sigma_pix=smooth_sigma_pix
#     )
#     if _HAS_SKIMAGE:
#         # 0 等高線を抽出
#         contours = find_contours(detA, 0.0)  # list of (N,2) in (row[y], col[x])
#         return contours
#     else:
#         # skimage が無い場合は簡易ゼロ交差抽出（8近傍）
#         det = detA
#         s = np.sign(det)
#         zc = (s[:, 1:] * s[:, :-1] < 0) | (s[1:, :] * s[:-, :] < 0)
#         # 画素中心の近似点を返す（粗い）
#         yy_idx, xx_idx = np.where(
#             np.pad(zc, ((0,1),(0,1)), constant_values=False)
#         )
#         return [np.stack([yy_idx, xx_idx], axis=1)]
        
# def caustics_from_critical(contours, xx, yy, *,
#                            x0, y0, b, q, pa, gamma, theta_gamma, kappa0=0.0):
#     """
#     クリティカル曲線上の各点 θ をレンズ方程式 β=θ-α(θ) で写像し、
#     カスプ/カットを含むカウスティックの離散点列を返す。
#     """
#     ax, ay = deflection_SIE_plus_ES(xx=xx, yy=yy, x0=x0, y0=y0, b=b, q=q, pa=pa,
#                                     gamma=gamma, theta_gamma=theta_gamma, kappa=kappa0)
#     caustics = []
#     for C in contours:
#         # C は (N,2) で (y,x) の連続座標（画素単位）
#         y = C[:, 0]; x = C[:, 1]
#         # 最近傍ピクセルに丸めてサンプル（双一次補間にしたければ map_coordinates を使用）
#         yi = np.clip(np.rint(y).astype(int), 0, ay.shape[0]-1)
#         xi = np.clip(np.rint(x).astype(int), 0, ax.shape[1]-1)
#         theta_x = (xi - (ax.shape[1]-1)/2) * (xx[0,1]-xx[0,0])  # xx は等間隔前提
#         theta_y = (yi - (ay.shape[0]-1)/2) * (yy[1,0]-yy[0,0])
#         beta_x = theta_x - ax[yi, xi]
#         beta_y = theta_y - ay[yi, xi]
#         caustics.append(np.stack([beta_y, beta_x], axis=1))  # (yβ, xβ)
#     return caustics