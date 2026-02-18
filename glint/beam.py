import numpy as np

def _normalize_kernel(kernel):
    """Normalize a 2D kernel so that its peak is 1."""
    kernel_max = np.max(kernel)
    if kernel_max > 0:
        return kernel / kernel_max
    else:
        raise ValueError("Kernel max is zero or negative, cannot normalize.")


def cleanbeam_from_header(npix, header):
    """
    正方形 (npix, npix) の復元ビームPSFを生成。npixは奇数とする。
    - BMAJ, BMIN : FWHM [deg]
    - BPA        : [deg], 北方向が0度、東方向が90度

    戻り値: (npix, npix) の正規化PSF
    """
    if npix % 2 == 0:
        raise ValueError("npix should be odd to center the beam at a pixel.")

    # FWHM → σ [pixel]
    BMAJ = header['BMAJ']
    BMIN = header['BMIN']
    BPA  = header['BPA']
    pixscale = np.abs(header['CDELT2']) 
    
    FWHM_x_pix = BMAJ / pixscale
    FWHM_y_pix = BMIN / pixscale
    sx_pix = FWHM_x_pix / (2.0*np.sqrt(2*np.log(2)))
    sy_pix = FWHM_y_pix / (2.0*np.sqrt(2*np.log(2)))

    # FITS BPA(北基準) → NumPy rotation(東基準)
    theta = np.deg2rad(90.0 + BPA)
    c, s = np.cos(theta), np.sin(theta)

    yy, xx = np.indices((npix, npix))
    cy, cx = (npix-1)/2.0, (npix-1)/2.0
    x = xx - cx
    y = yy - cy

    # 回転（major軸が x'）
    xp =  c*x - s*y
    yp =  s*x + c*y

    k = np.exp(-0.5*((xp/sx_pix)**2 + (yp/sy_pix)**2))
    return _normalize_kernel(k)