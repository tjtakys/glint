"""
lensing.py : Calculate lensing deflection angles and related quantities.

Models
------
- SIE (Singular Isothermal Ellipsoid): implemented.
"""

from __future__ import annotations
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
    # x0_pix = (nx-1)/2.0
    # y0_pix = (ny-1)/2.0
    x0_pix = nx/2.0
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
    x0_src_pix = (nx_src - 1) / 2.0
    y0_src_pix = (ny_src - 1) / 2.0

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
"""
det A(theta) = 0 となる theta がcritical lines、causticsはそれをsource planeに戻せばいい。
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
    x0_pix = (nx - 1) / 2.0
    y0_pix = (ny - 1) / 2.0

    x_pix = (curve_xy[:, 0] - x0_arcsec) / pixscale_arcsec + x0_pix
    y_pix = (curve_xy[:, 1] - y0_arcsec) / pixscale_arcsec + y0_pix

    return np.column_stack([x_pix, y_pix])


def compute_critical_lines_and_caustics(ctx, deflector, lens_params,
                                        min_points=10, min_length=0.0):
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
    _, _, _, _, detA, mu = jacobian_lens_mapping(ctx.xx_img, ctx.yy_img, alpha_x, alpha_y)

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