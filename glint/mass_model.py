"""Circular-velocity models for physical mass components.

All radii are in kpc, masses are in solar masses, and returned squared
velocities are in ``(km/s)^2``.  Keeping these models here separates physical
mass distributions from the line-emission models in :mod:`glint.source`.

The disk, bulge, and NFW implementations consolidate the experimental work
from commit ``1f9df58`` on ``experiment/mass-forward-modeling``.  The formulas
here use one explicit unit convention, robust zero-radius handling, and the
standard exponential-disk variable ``y = R / (2 R_d)``.

These functions are fitting-time numerical kernels. Callers must pass a
finite, non-negative floating-point NumPy array as ``radius_kpc`` and
pre-validated physical scalar parameters; validation is intentionally not
repeated here.
"""

from __future__ import annotations

import numpy as np
from astropy.constants import G
import astropy.units as u
from scipy.special import gammainc, i0e, i1e, k0e, k1e


G_KPC_KMS2_MSUN = G.to_value(u.kpc * (u.km / u.s) ** 2 / u.Msun)


def r200_kpc_from_m200(
    mass_200_msun: float,
    critical_density_msun_kpc3: float,
) -> float:
    """Return the radius enclosing 200 times the critical density [kpc].
    ちなみに、unit使うと計算770倍遅くなったので注意"""
    return (3.0 * mass_200_msun / (800.0 * np.pi * critical_density_msun_kpc3)) ** (1.0 / 3.0)


def vcirc2_exponential_disk(
    radius_kpc,
    mass_msun: float,
    scale_radius_kpc: float,
) -> np.ndarray:
    """Squared circular speed of a razor-thin exponential disk.

    This is the Freeman-disk expression

        V_c^2(R) = 2 G M_d R_d^{-1} y^2
                   [I_0(y)K_0(y)-I_1(y)K_1(y)]

    where ``y = R / (2 R_d)``. The exponentially scaled Bessel functions are
    numerically stable at large ``y`` and faster than generic-order wrappers.
    """
    flat_radius = radius_kpc.reshape(-1)
    flat_velocity2 = np.zeros_like(flat_radius)
    positive = flat_radius > 0.0
    if np.any(positive):
        y = flat_radius[positive] / (2.0 * scale_radius_kpc)
        bessel_term = i0e(y) * k0e(y) - i1e(y) * k1e(y)
        flat_velocity2[positive] = (
            2.0 * G_KPC_KMS2_MSUN * mass_msun / scale_radius_kpc * y**2 * bessel_term
        )
    return flat_velocity2.reshape(radius_kpc.shape)


def vcirc2_hernquist_bulge(
    radius_kpc,
    mass_msun: float,
    effective_radius_kpc: float,
) -> np.ndarray:
    """Squared circular speed of a spherical Hernquist bulge.

    ``effective_radius_kpc`` is the projected half-light radius.  It is
    converted to the Hernquist scale radius using ``R_e = 1.8153 a``.
    """
    scale_radius_kpc = effective_radius_kpc / 1.8153
    return G_KPC_KMS2_MSUN * mass_msun * radius_kpc / (radius_kpc + scale_radius_kpc) ** 2


def vcirc2_sersic_bulge(
    radius_kpc,
    mass_msun: float,
    effective_radius_kpc: float,
    sersic_index: float,
) -> np.ndarray:
    """Squared circular speed of a spherical Sérsic bulge.

    The projected Sérsic profile is deprojected with the Prugniel--Simien
    approximation. Its enclosed mass has the analytic form:

        M(<r) = M_tot P(n(3-p), b_n (r/R_e)^(1/n))

    where ``P`` is the regularized lower incomplete gamma function.  The
    approximations for ``b_n`` and ``p`` are intended for ``n > 0.36`` and
    ``0.6 < n < 10``, respectively.
    """
    # Prugniel & Simien (1997), A&A, 321, 111
    b_n = 2.0 * sersic_index - 1.0 / 3.0 + 0.009876 / sersic_index
    # Márquez et al. (2000), A&A, 353, 873
    p = 1.0 - 0.6097 / sersic_index + 0.05563 / sersic_index**2
    gamma_shape = sersic_index * (3.0 - p)
    gamma_argument = b_n * (radius_kpc / effective_radius_kpc) ** (1.0 / sersic_index)
    enclosed_mass = mass_msun * gammainc(gamma_shape, gamma_argument) # scipyのgammaincは正規化されている
    return np.divide(
        G_KPC_KMS2_MSUN * enclosed_mass,
        radius_kpc,
        out=np.zeros_like(radius_kpc),
        where=radius_kpc > 0.0,
    )


def vcirc2_disk_bulge(
    radius_kpc,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
) -> np.ndarray:
    """Squared circular speed of an exponential disk plus Hernquist bulge."""
    return vcirc2_exponential_disk(
        radius_kpc,
        mass_msun=disk_mass_msun,
        scale_radius_kpc=disk_scale_radius_kpc,
    ) + vcirc2_hernquist_bulge(
        radius_kpc,
        mass_msun=bulge_mass_msun,
        effective_radius_kpc=bulge_effective_radius_kpc,
    )


def vcirc2_nfw_halo(
    radius_kpc,
    mass_200_msun: float,
    radius_200_kpc: float,
    concentration_200: float,
) -> np.ndarray:
    """Squared circular speed of a spherical NFW halo.

    ``mass_200_msun`` is the mass enclosed by ``radius_200_kpc`` and
    ``concentration_200 = radius_200_kpc / r_s``.  Cosmology is intentionally
    not hidden inside this function; callers must supply a consistent virial
    mass and radius.
    """
    scale_radius_kpc = radius_200_kpc / concentration_200
    x = radius_kpc / scale_radius_kpc
    enclosed_shape = np.log1p(x) - x / (1.0 + x)
    normalization = np.log1p(concentration_200) - concentration_200 / (1.0 + concentration_200)
    enclosed_mass = mass_200_msun * enclosed_shape / normalization
    return np.divide(
        G_KPC_KMS2_MSUN * enclosed_mass,
        radius_kpc,
        out=np.zeros_like(radius_kpc),
        where=radius_kpc > 0.0,
    )


def vcirc2_disk_bulge_halo(
    radius_kpc,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
    halo_mass_200_msun: float,
    halo_radius_200_kpc: float,
    halo_concentration_200: float,
) -> np.ndarray:
    """Squared circular speed of disk, bulge, and NFW halo components."""
    baryon_velocity2 = vcirc2_disk_bulge(
        radius_kpc,
        disk_mass_msun=disk_mass_msun,
        disk_scale_radius_kpc=disk_scale_radius_kpc,
        bulge_mass_msun=bulge_mass_msun,
        bulge_effective_radius_kpc=bulge_effective_radius_kpc,
    )
    halo_velocity2 = vcirc2_nfw_halo(
        radius_kpc,
        mass_200_msun=halo_mass_200_msun,
        radius_200_kpc=halo_radius_200_kpc,
        concentration_200=halo_concentration_200,
    )
    return baryon_velocity2 + halo_velocity2
