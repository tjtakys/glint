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
from scipy.special import i0e, i1e, k0e, k1e


G_KPC_KMS2_MSUN = G.to_value(u.kpc * (u.km / u.s) ** 2 / u.Msun)
HERNQUIST_RE_OVER_A = 1.8153


def exponential_disk_vcirc2(
    radius_kpc,
    mass_msun: float,
    scale_radius_kpc: float,
) -> np.ndarray:
    """Squared circular speed of a razor-thin exponential disk.

    This is the Freeman-disk expression

        V_c^2(R) = 2 G M_d R_d^{-1} y^2 [I_0(y)K_0(y)-I_1(y)K_1(y)], (J. Binney & S. Tremaine 2008)

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


def hernquist_bulge_vcirc2(
    radius_kpc,
    mass_msun: float,
    effective_radius_kpc: float,
) -> np.ndarray:
    """Squared circular speed of a spherical Hernquist bulge.

    ``effective_radius_kpc`` is the projected half-light radius.  It is
    converted to the Hernquist scale radius using ``R_e = 1.8153 a``.
    """
    scale_radius_kpc = effective_radius_kpc / HERNQUIST_RE_OVER_A
    return (
        G_KPC_KMS2_MSUN
        * mass_msun
        * radius_kpc
        / (radius_kpc + scale_radius_kpc) ** 2
    )


def disk_plus_bulge_vcirc2(
    radius_kpc,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
) -> np.ndarray:
    """Squared circular speed of an exponential disk plus Hernquist bulge."""
    return exponential_disk_vcirc2(
        radius_kpc,
        mass_msun=disk_mass_msun,
        scale_radius_kpc=disk_scale_radius_kpc,
    ) + hernquist_bulge_vcirc2(
        radius_kpc,
        mass_msun=bulge_mass_msun,
        effective_radius_kpc=bulge_effective_radius_kpc,
    )


def disk_plus_bulge_vcirc(
    radius_kpc,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
) -> np.ndarray:
    """Circular speed of an exponential disk plus Hernquist bulge [km/s]."""
    velocity2 = disk_plus_bulge_vcirc2(
        radius_kpc,
        disk_mass_msun=disk_mass_msun,
        disk_scale_radius_kpc=disk_scale_radius_kpc,
        bulge_mass_msun=bulge_mass_msun,
        bulge_effective_radius_kpc=bulge_effective_radius_kpc,
    )
    return np.sqrt(np.maximum(velocity2, 0.0))


def disk_plus_bulge_vcirc_components(
    radius_kpc,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disk, bulge, and total circular speeds [km/s]."""
    disk_velocity2 = exponential_disk_vcirc2(
        radius_kpc,
        mass_msun=disk_mass_msun,
        scale_radius_kpc=disk_scale_radius_kpc,
    )
    bulge_velocity2 = hernquist_bulge_vcirc2(
        radius_kpc,
        mass_msun=bulge_mass_msun,
        effective_radius_kpc=bulge_effective_radius_kpc,
    )
    return (
        np.sqrt(np.maximum(disk_velocity2, 0.0)),
        np.sqrt(np.maximum(bulge_velocity2, 0.0)),
        np.sqrt(np.maximum(disk_velocity2 + bulge_velocity2, 0.0)),
    )


def nfw_halo_vcirc2(
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
    normalization = (
        np.log1p(concentration_200)
        - concentration_200 / (1.0 + concentration_200)
    )
    enclosed_mass = mass_200_msun * enclosed_shape / normalization
    return np.divide(
        G_KPC_KMS2_MSUN * enclosed_mass,
        radius_kpc,
        out=np.zeros_like(radius_kpc),
        where=radius_kpc > 0.0,
    )


def disk_bulge_halo_vcirc2(
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
    baryon_velocity2 = disk_plus_bulge_vcirc2(
        radius_kpc,
        disk_mass_msun=disk_mass_msun,
        disk_scale_radius_kpc=disk_scale_radius_kpc,
        bulge_mass_msun=bulge_mass_msun,
        bulge_effective_radius_kpc=bulge_effective_radius_kpc,
    )
    halo_velocity2 = nfw_halo_vcirc2(
        radius_kpc,
        mass_200_msun=halo_mass_200_msun,
        radius_200_kpc=halo_radius_200_kpc,
        concentration_200=halo_concentration_200,
    )
    return baryon_velocity2 + halo_velocity2


def disk_bulge_halo_vcirc(
    radius_kpc,
    disk_mass_msun: float,
    disk_scale_radius_kpc: float,
    bulge_mass_msun: float,
    bulge_effective_radius_kpc: float,
    halo_mass_200_msun: float,
    halo_radius_200_kpc: float,
    halo_concentration_200: float,
) -> np.ndarray:
    """Circular speed of disk, bulge, and NFW halo components [km/s]."""
    velocity2 = disk_bulge_halo_vcirc2(
        radius_kpc,
        disk_mass_msun=disk_mass_msun,
        disk_scale_radius_kpc=disk_scale_radius_kpc,
        bulge_mass_msun=bulge_mass_msun,
        bulge_effective_radius_kpc=bulge_effective_radius_kpc,
        halo_mass_200_msun=halo_mass_200_msun,
        halo_radius_200_kpc=halo_radius_200_kpc,
        halo_concentration_200=halo_concentration_200,
    )
    return np.sqrt(np.maximum(velocity2, 0.0))
