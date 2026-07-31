import numpy as np

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def _gaussian_beam(npix, sigma_major_pix, sigma_minor_pix, position_angle_deg):
    """
    畳み込む前の単位： Jy/pix から 畳み込んだ後 Jy/beam への変換のために、beamをpeak=1に規格化する。
    """
    if npix <= 0 or npix % 2 == 0:
        raise ValueError("npix must be a positive odd integer")

    coordinate = np.arange(npix, dtype=float) - (npix - 1) / 2
    xx, yy = np.meshgrid(coordinate, coordinate)
    angle = np.deg2rad(position_angle_deg)
    # 受動回転
    x_major =  np.cos(angle) * xx + np.sin(angle) * yy
    y_minor = -np.sin(angle) * xx + np.cos(angle) * yy
    beam = np.exp(-0.5 * ((x_major / sigma_major_pix) ** 2 + (y_minor / sigma_minor_pix) ** 2))
    return beam / beam.max()


def cleanbeam_from_header(npix, header):
    """
    npixは奇数とする。
    - BMAJ, BMIN : FWHM [deg]
    - BPA        : [deg], 北方向が0度、東方向が90度
    peak = 1 に規格化
    """
    pixel_size_deg = abs(header["CDELT2"])
    sigma_major_pix = header["BMAJ"] / pixel_size_deg * FWHM_TO_SIGMA
    sigma_minor_pix = header["BMIN"] / pixel_size_deg * FWHM_TO_SIGMA

    # FITS BPA is measured from north through east; array angle is from +x.
    position_angle_deg = 90 + header["BPA"]
    return _gaussian_beam(npix, sigma_major_pix, sigma_minor_pix, position_angle_deg)



def gaussian_psf_kernel(
    pixel_size_arcsec,
    fwhm_major_arcsec,
    fwhm_minor_arcsec,
    position_angle_deg=0,
    truncate_sigma=4,
):
    """Create a peak-normalized Gaussian beam from angular FWHM values."""
    sigma_major_pix = fwhm_major_arcsec / pixel_size_arcsec * FWHM_TO_SIGMA
    sigma_minor_pix = fwhm_minor_arcsec / pixel_size_arcsec * FWHM_TO_SIGMA
    half_size = int(np.ceil(truncate_sigma * max(sigma_major_pix, sigma_minor_pix)))

    return _gaussian_beam(2 * half_size + 1,sigma_major_pix,sigma_minor_pix,position_angle_deg)