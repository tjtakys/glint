"""Compute galaxy circular-velocity profiles from physical mass distributions.

Implemented mass models:
- Razor-thin exponential disk
- Spherical Hernquist bulge
- Spherical Sérsic bulge
- Spherical NFW dark-matter halo

Input radii are in kpc, input masses are in solar masses, and returned squared circular velocities are in (km/s)^2.
"""


import numpy as np
from astropy.constants import G
import astropy.units as u
from scipy.special import gammainc, i0e, i1e, k0e, k1e


G_KPC_KMS2_MSUN = G.to_value(u.kpc * (u.km / u.s) ** 2 / u.Msun)


#--------------------------
# Disk
#--------------------------

def vcirc2_exponential_disk(
    radius_kpc: np.ndarray,
    mass_msun: float,
    scale_radius_kpc: float, # 半質量半径 Re ではないことに注意
) -> np.ndarray:
    """Squared circular velocity of a razor-thin exponential disk.
    (J. Binney & S. Tremaine 2008, Galactic Dynamics, 2nd ed., eq. 2.165)

    scale_radius_kpc Rd は 投影した2Dディスクのスケール長（1/eに落ちる半径）であり、半質量半径 Re ではないことに注意。半質量半径は Re = 1.67835 Rd である。Rdに含まれるのは26.4%の質量だけ。
    I0 K0 - I1 K1 の各項は 修正ベッセル関数だが、大きな引数では I0~e^x, K0~e^-x となるので、一応オーバーフローを避けるために scipy.special.i0e, i1e, k0e, k1e を使う。掛け算するとexpは相殺するので同じ。速度は変わらない。
    y=0の場合、Kが+infに発散してしまうので場合分け必要
    なお、radius=0での除算を回避するだけなら、下の vcirc2_sersic_bulge関数のようにnp.divideを使った方が高速だが、ベッセル関数が0で発散するので、ここでは場合分けしている。
    """
    vcirc2 = np.zeros_like(radius_kpc, dtype=float)
    positive = radius_kpc > 0.0
    y = radius_kpc[positive] / (2.0 * scale_radius_kpc)
    bessel_term = i0e(y) * k0e(y) - i1e(y) * k1e(y)
    vcirc2[positive] = 2.0 * G_KPC_KMS2_MSUN * mass_msun / scale_radius_kpc * y**2 * bessel_term
    return vcirc2



#--------------------------
# Bulge
#--------------------------

def vcirc2_hernquist_bulge(
    radius_kpc: np.ndarray,
    mass_msun: float,
    effective_radius_kpc: float, # projected half-mass radius
) -> np.ndarray:
    """Squared circular velocity of a spherical Hernquist bulge.
    (Hernquist 1990, ApJ, 356, 359, eq. 16, 38)

    球対称バルジで、n=4のSersic profileの近似。
    r<<a では V_circ^2 ~ r, r>>a では V_circ^2 ~ 1/r となり Keplerianに落ちる。
    Effective radius R_e は projected half-mass radius であり、あくまで2Dの半質量半径であることに注意。
    この場合、R_e = 1.8153 a  である。
    memo: Hernquist scale radius a
    - r=aの内側に全質量の25%が含まれる
    - V_circは r=aで最大値をとる
    """
    if mass_msun == 0.0: # bulgeなしモデルでもこの関数を呼ぶので
        return np.zeros_like(radius_kpc, dtype=float)
    scale_radius_kpc = effective_radius_kpc / 1.8153
    vcirc2 = G_KPC_KMS2_MSUN * mass_msun * radius_kpc / (radius_kpc + scale_radius_kpc) ** 2
    return vcirc2


def vcirc2_sersic_bulge(
    radius_kpc: np.ndarray,
    mass_msun: float,
    effective_radius_kpc: float, # projected half-mass radius
    sersic_index: float,
) -> np.ndarray:
    """Squared circular velocity of a spherical Sérsic bulge using the Prugniel--Simien approximation.
    Effective radius R_e は projected half-mass radius であり、あくまで2Dの半質量半径であることに注意。
    pの近似式は 0.6 < n < 10 で良い。
    """
    if mass_msun == 0.0: # bulgeなしモデルでもこの関数を呼ぶので
        return np.zeros_like(radius_kpc, dtype=float)
    
    # Terzic & Graham (2005), MNRAS, 362, 197, eq. A2, A3
    b_n = 2.0 * sersic_index - 1.0 / 3.0 + 0.009876 / sersic_index # Prugniel & Simien (1997), A&A, 321, 111, eq. A3a
    # Márquez et al. (2000), A&A, 353, 873
    p = 1.0 - 0.6097 / sersic_index + 0.05563 / sersic_index**2 # https://articles.adsabs.harvard.edu/pdf/2000A%26A...353..873M
    gamma_shape = sersic_index * (3.0 - p)
    gamma_argument = b_n * (radius_kpc / effective_radius_kpc) ** (1.0 / sersic_index)
    enclosed_mass = mass_msun * gammainc(gamma_shape, gamma_argument) # scipyのgammaincは正規化されている
    return np.divide( # ここではradius=0での除算を回避したいだけなので、この書き方の方が場合分けするより高速
        G_KPC_KMS2_MSUN * enclosed_mass,
        radius_kpc,
        out=np.zeros_like(radius_kpc, dtype=float),
        where=radius_kpc > 0.0,
    )



#--------------------------
# DM halo
#--------------------------

def r200_kpc_from_m200(
    mass_200_msun: float,
    critical_density_msun_kpc3: float,
) -> float:
    """Return the radius within which the mean density is 200 times the critical density [kpc].
    ちなみに、unit使うと計算770倍遅くなったので注意"""
    return (3.0 * mass_200_msun / (800.0 * np.pi * critical_density_msun_kpc3)) ** (1.0 / 3.0)

def vcirc2_nfw_halo(
    radius_kpc: np.ndarray,
    mass_200_msun: float,
    radius_200_kpc: float,
    concentration_200: float,
) -> np.ndarray:
    """Squared circular velocity of a spherical NFW halo.
    Navarro, Frenk & White (1997), ApJ, 490, 493, eq. 3
    """
    vcirc2_200 = G_KPC_KMS2_MSUN * mass_200_msun / radius_200_kpc
    x = radius_kpc / radius_200_kpc
    cx = concentration_200 * x
    enclosed_mass_shape = np.log1p(cx) - cx / (1.0 + cx) # np.log1p(x) の方が np.log(1+x)よりも x<<1 のときの精度が良い
    total_mass_shape = np.log1p(concentration_200) - concentration_200 / (1.0 + concentration_200)
    return np.divide(
        vcirc2_200 * enclosed_mass_shape,
        x * total_mass_shape,
        out=np.zeros_like(radius_kpc, dtype=float),
        where=x > 0.0,
    )


#--------------------------
# Total circular speed
#--------------------------

def vcirc2_disk_bulge(
    radius_kpc: np.ndarray,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
) -> np.ndarray:
    """Squared circular velocity of an exponential disk plus Hernquist bulge."""
    return vcirc2_exponential_disk(
        radius_kpc,
        mass_msun=disk_mass_msun,
        scale_radius_kpc=disk_scale_radius_kpc,
    ) + vcirc2_hernquist_bulge(
        radius_kpc,
        mass_msun=bulge_mass_msun,
        effective_radius_kpc=bulge_effective_radius_kpc,
    )


def vcirc2_disk_bulge_halo(
    radius_kpc: np.ndarray,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
    halo_mass_200_msun: float,
    halo_radius_200_kpc: float,
    halo_concentration_200: float,
) -> np.ndarray:
    """Squared circular speed of disk, bulge, and halo components."""
    return vcirc2_disk_bulge(
        radius_kpc,
        disk_mass_msun=disk_mass_msun,
        disk_scale_radius_kpc=disk_scale_radius_kpc,
        bulge_mass_msun=bulge_mass_msun,
        bulge_effective_radius_kpc=bulge_effective_radius_kpc,
    ) + vcirc2_nfw_halo(
        radius_kpc,
        mass_200_msun=halo_mass_200_msun,
        radius_200_kpc=halo_radius_200_kpc,
        concentration_200=halo_concentration_200,
    )