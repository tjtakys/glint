from dataclasses import dataclass
from os import PathLike

import numpy as np
from astropy.constants import c as cspeed
# import numba as nb
from finufft import nufft2d1, nufft2d2, nufft2d3

ARCSEC2RAD = np.deg2rad(1/3600)


@dataclass(frozen=True, slots=True)
class ProcessedUVData:
    """Validated visibility data loaded from a processed UV ``.npz`` file.

    Array shapes follow the convention used by the GLINT notebooks:
    ``uvw_m`` is ``(3, nrow)``, ``uvw_lam`` is ``(3, nchan, nrow)``, and
    ``data``, ``sigma``, and ``flag`` are ``(nchan, nrow)``.
    """

    uvw_m: np.ndarray            # (3, nrow): (u, v, w) in meters
    uvw_lam: np.ndarray          # (3, nchan, nrow): (u, v, w) in wavelengths
    data: np.ndarray             # (nchan, nrow): complex visibilities
    sigma: np.ndarray            # (nchan, nrow): visibility uncertainties
    flag: np.ndarray             # (nchan, nrow): True for flagged visibilities
    freqs_hz: np.ndarray         # (nchan,): channel frequencies in Hz
    v_kms: np.ndarray            # (nchan,): channel velocities in km/s
    pb_fwhm_as: np.ndarray       # () or (nchan,): primary-beam FWHM in arcsec
    phase_center_deg: np.ndarray # (2,): (RA, Dec) phase center in degrees

    @property
    def nchan(self) -> int:
        return int(self.data.shape[0])

    @property
    def sigma_inv(self) -> np.ndarray:
        result = np.zeros_like(self.sigma, dtype=float)
        return np.divide(1.0, self.sigma, out=result, where=self.sigma > 0) # もしsigma<=0だと重み0

    @property
    def spec_res_sigma_kms(self) -> float:
        if self.nchan < 2:
            raise ValueError("At least two velocity channels are required to infer spectral resolution.")
        return float(abs(np.median(np.diff(self.v_kms))) / 2.355) # FWHM -> sigma

    def minimum_baseline_m(
        self,
        max_angular_scale_arcsec: float,
        *,
        scale_factor: float = 0.6,  # 経験的な値：https://almascience.nrao.edu/about-alma/alma-basics
    ) -> np.ndarray:
        """天体より明らかに大きいスケールを除くための最小baseline [m]を返す。"""
        if max_angular_scale_arcsec <= 0:
            raise ValueError("max_angular_scale_arcsec must be > 0.")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be > 0.")
        theta_rad = max_angular_scale_arcsec * ARCSEC2RAD
        wavelength_m = cspeed.value / self.freqs_hz  # (nchan,)
        return scale_factor * wavelength_m / theta_rad  # (nchan,)

    def flags_with_max_angular_scale_cut(
        self,
        max_angular_scale_arcsec: float,
        *,
        scale_factor: float = 0.6,
    ) -> np.ndarray:
        """天体より明らかに大きいスケールに対応するvisibilityをフラグする。"""
        minimum_baseline_m = self.minimum_baseline_m(  # (nchan,)
            max_angular_scale_arcsec,
            scale_factor=scale_factor,
        )
        uv_radius_m = np.hypot(self.uvw_m[0], self.uvw_m[1])  # (nrow,)
        return self.flag | (uv_radius_m[None, :] < minimum_baseline_m[:, None])  # (nchan, nrow)


def load_processed_uv(path: str | PathLike[str]) -> ProcessedUVData:
    """Load and validate a processed UV archive produced by the GLINT workflow."""
    required = {
        "uvw",
        "uvw_lam",
        "data",
        "sigma",
        "flag",
        "freqs",
        "v_kms",
        "pb_fwhm",
        "phase_dir",
    }
    with np.load(path, allow_pickle=False) as uvpack:
        missing = required.difference(uvpack.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"Processed UV archive is missing required fields: {names}")
        uvw_m = np.asarray(uvpack["uvw"], dtype=float)
        uvw_lam = np.asarray(uvpack["uvw_lam"], dtype=float)
        data = np.asarray(uvpack["data"])
        sigma = np.asarray(uvpack["sigma"], dtype=float)
        flag = np.asarray(uvpack["flag"], dtype=bool)
        freqs_hz = np.atleast_1d(np.asarray(uvpack["freqs"], dtype=float))
        v_kms = np.atleast_1d(np.asarray(uvpack["v_kms"], dtype=float))
        pb_fwhm_as = np.asarray(uvpack["pb_fwhm"], dtype=float)
        phase_center_deg = np.asarray(uvpack["phase_dir"], dtype=float)

    if data.ndim != 2:
        raise ValueError(f"data must have shape (nchan, nrow); got {data.shape}.")
    nchan, nrow = data.shape
    if uvw_m.shape != (3, nrow):
        raise ValueError(f"uvw must have shape (3, {nrow}); got {uvw_m.shape}.")
    if uvw_lam.shape != (3, nchan, nrow):
        raise ValueError(
            f"uvw_lam must have shape (3, {nchan}, {nrow}); got {uvw_lam.shape}."
        )
    for name, array in (("sigma", sigma), ("flag", flag)):
        if array.shape != data.shape:
            raise ValueError(f"{name} must have shape {data.shape}; got {array.shape}.")
    if freqs_hz.shape != (nchan,):
        raise ValueError(f"freqs must have shape ({nchan},); got {freqs_hz.shape}.")
    if v_kms.shape != (nchan,):
        raise ValueError(f"v_kms must have shape ({nchan},); got {v_kms.shape}.")
    if pb_fwhm_as.ndim > 1 or (pb_fwhm_as.ndim == 1 and pb_fwhm_as.shape != (nchan,)):
        raise ValueError("pb_fwhm must be scalar or have shape (nchan,).")
    if phase_center_deg.shape != (2,):
        raise ValueError(f"phase_dir must have shape (2,); got {phase_center_deg.shape}.")
    if np.any(freqs_hz <= 0):
        raise ValueError("All frequencies must be > 0 Hz.")
    if np.any((sigma <= 0) & ~flag):
        raise ValueError("Unflagged visibility sigma values must be > 0.")

    return ProcessedUVData(
        uvw_m=uvw_m,
        uvw_lam=uvw_lam,
        data=data,
        sigma=sigma,
        flag=flag,
        freqs_hz=freqs_hz,
        v_kms=v_kms,
        pb_fwhm_as=pb_fwhm_as,
        phase_center_deg=phase_center_deg,
    )


def get_msdata(
    path: str | PathLike[str],
    *,
    data_column: str = "DATA",
    spw_id: int = 0,
) -> dict[str, np.ndarray]:
    """Read one spectral window from a CASA Measurement Set without processing it.

    Returns arrays with shapes ``uvw_m=(3, nrow)``,
    ``data_pol=flag_pol=(npol, nchan, nrow)``,
    ``weight_pol=sigma_pol=(npol, nrow)``, ``freqs_hz=(nchan,)``,
    ``phase_center_rad=(2,)``, and all row metadata arrays ``(nrow,)``.
    """
    try:
        from casatools import table
    except ImportError as exc:
        raise ImportError("casatools is required to load a Measurement Set.") from exc

    ms_path = str(path)
    tb = table()
    try:
        tb.open(ms_path + "/DATA_DESCRIPTION")
        spw_by_data_desc = np.asarray(tb.getcol("SPECTRAL_WINDOW_ID"), dtype=int)
    finally:
        tb.close()
    matching_data_desc = np.flatnonzero(spw_by_data_desc == spw_id)
    if matching_data_desc.size == 0:
        raise ValueError(f"Spectral window {spw_id} is not present in the Measurement Set.")

    selected = None
    try:
        tb.open(ms_path)
        data_desc_id = np.asarray(tb.getcol("DATA_DESC_ID"), dtype=int)
        rows = np.flatnonzero(np.isin(data_desc_id, matching_data_desc))
        if rows.size == 0:
            raise ValueError(f"Spectral window {spw_id} has no rows in the Measurement Set.")
        selected = tb.selectrows(rows.tolist())
        uvw_m = np.asarray(selected.getcol("UVW"), dtype=float)
        data_pol = np.asarray(selected.getcol(data_column.upper()))
        weight_pol = np.asarray(selected.getcol("WEIGHT"), dtype=float)
        sigma_pol = np.asarray(selected.getcol("SIGMA"), dtype=float)
        flag_pol = np.asarray(selected.getcol("FLAG"), dtype=bool)
        antenna1 = np.asarray(selected.getcol("ANTENNA1"), dtype=int)
        antenna2 = np.asarray(selected.getcol("ANTENNA2"), dtype=int)
        time_s = np.asarray(selected.getcol("TIME"), dtype=float)
        scan_number = np.asarray(selected.getcol("SCAN_NUMBER"), dtype=int)
        field_id = np.asarray(selected.getcol("FIELD_ID"), dtype=int)
    finally:
        if selected is not None:
            selected.close()
        tb.close()

    unique_field_id = np.unique(field_id)
    if unique_field_id.size != 1:
        raise ValueError(
            "Selected Measurement Set rows must belong to exactly one field; "
            f"got field IDs {unique_field_id.tolist()}."
        )

    try:
        tb.open(ms_path + "/SPECTRAL_WINDOW")
        freqs_hz = np.atleast_1d(
            np.asarray(tb.getcell("CHAN_FREQ", int(spw_id)), dtype=float)
        )
    finally:
        tb.close()
    try:
        tb.open(ms_path + "/FIELD")
        phase_center_rad = np.asarray(
            tb.getcell("PHASE_DIR", int(unique_field_id[0])), dtype=float
        ).squeeze()
    finally:
        tb.close()
    if phase_center_rad.shape != (2,):
        raise ValueError(f"PHASE_DIR must contain two coordinates; got {phase_center_rad.shape}.")

    return {
        "uvw_m": uvw_m,
        "data_pol": data_pol,
        "weight_pol": weight_pol,
        "sigma_pol": sigma_pol,
        "flag_pol": flag_pol,
        "freqs_hz": freqs_hz,
        "phase_center_rad": phase_center_rad,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "time_s": time_s,
        "scan_number": scan_number,
        "field_id": field_id,
    }


def process_msdata(
    msdata: dict[str, np.ndarray],
    *,
    dish_diameter_m: float = 12.0,
) -> dict[str, np.ndarray]:
    """Flag, polarization-average, and derive coordinates from raw MS data.

    Returns ``uvw_m=(3, nrow)``, ``uvw_lam=(3, nchan, nrow)``,
    ``data=sigma=weight=flag=(nchan, nrow)``,
    ``freqs_hz=pb_fwhm_as=(nchan,)``, and ``phase_center_deg=(2,)``.
    """
    if dish_diameter_m <= 0:
        raise ValueError("dish_diameter_m must be > 0.")

    uvw_m = np.asarray(msdata["uvw_m"], dtype=float)
    data_pol = np.asarray(msdata["data_pol"])
    weight_pol = np.asarray(msdata["weight_pol"], dtype=float)
    flag_pol = np.asarray(msdata["flag_pol"], dtype=bool)
    freqs_hz = np.atleast_1d(np.asarray(msdata["freqs_hz"], dtype=float))
    phase_center_rad = np.asarray(msdata["phase_center_rad"], dtype=float)

    if data_pol.ndim == 2:
        data_pol = data_pol[:, None, :]
    if flag_pol.ndim == 2:
        flag_pol = flag_pol[:, None, :]
    if data_pol.ndim != 3 or flag_pol.shape != data_pol.shape:
        raise ValueError(
            "DATA and FLAG must have shape (npol, nchan, nrow); "
            f"got {data_pol.shape} and {flag_pol.shape}."
        )
    npol, nchan, nrow = data_pol.shape
    if uvw_m.shape != (3, nrow):
        raise ValueError(f"UVW must have shape (3, {nrow}); got {uvw_m.shape}.")
    if weight_pol.shape == (npol, nrow):
        weight_pol = np.broadcast_to(weight_pol[:, None, :], data_pol.shape)
    elif weight_pol.shape != data_pol.shape:
        raise ValueError(
            "WEIGHT must have shape (npol, nrow) or match DATA; "
            f"got {weight_pol.shape}."
        )
    if freqs_hz.shape != (nchan,):
        raise ValueError(
            f"CHAN_FREQ has shape {freqs_hz.shape}, but DATA has {nchan} channels."
        )
    if phase_center_rad.shape != (2,):
        raise ValueError(f"PHASE_DIR must contain two coordinates; got {phase_center_rad.shape}.")

    effective_weight = np.where(flag_pol, 0.0, weight_pol)
    weight = np.sum(effective_weight, axis=0)
    numerator = np.sum(effective_weight * data_pol, axis=0)
    data = np.divide(
        numerator,
        weight,
        out=np.zeros((nchan, nrow), dtype=data_pol.dtype),
        where=weight > 0,
    )
    flag = weight <= 0
    sigma = np.full((nchan, nrow), np.inf, dtype=float)
    sigma[~flag] = 1.0 / np.sqrt(weight[~flag])

    wavelength_m = cspeed.value / freqs_hz
    uvw_lam = uvw_m[:, None, :] / wavelength_m[None, :, None]
    pb_fwhm_as = 1.13 * wavelength_m / dish_diameter_m / ARCSEC2RAD

    return {
        "uvw_m": uvw_m,
        "uvw_lam": uvw_lam,
        "data": data,
        "sigma": sigma,
        "weight": weight,
        "flag": flag,
        "freqs_hz": freqs_hz,
        "pb_fwhm_as": pb_fwhm_as,
        "phase_center_deg": np.rad2deg(phase_center_rad),
    }

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
