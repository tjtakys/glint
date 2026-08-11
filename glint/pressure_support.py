"""Pressure support for exponential gas profiles."""

from __future__ import annotations
from typing import Literal
import numpy as np


VerticalModel = Literal["constant_scale_height", "self_gravitating_hydrostatic"]


class UnphysicalPressureSupportError(ValueError):
    """pressure supportが重力より大きくなったらエラーにして-np.nanを返してサンプリングから外すようにする"""


def pressure_supported_rotation_velocity(
    radius: np.ndarray,
    circular_velocity_kms: np.ndarray,
    radial_velocity_dispersion_kms: np.ndarray,
    surface_density_scale_radius: float,
    velocity_dispersion_scale_radius: float | None = None,
    *,
    vertical_model: VerticalModel = "constant_scale_height",
) -> np.ndarray:
    """指数関数profileにpressure supportを適用した回転速度を計算。
    圧力項は、sigma_R^2 * d ln(rho_R sigma_R^2) / d ln(R)
    (e.g., https://ui.adsabs.harvard.edu/abs/2022A%26A...658A..76B/abstract Appe.A)

    surface_density(R) ∝ exp(-R / R_surface)
    sigma_R(R)         ∝ exp(-R / R_sigma)
    どちらも指数関数を仮定して計算

    constant_scale_height:
        v_rot**2 = v_circ**2 - sigma_R**2 * R * (1 / R_surface + 2 / R_sigma)
    self_gravitating_hydrostatic:
        v_rot**2 = v_circ**2 - 2 * sigma_R**2 * R / R_surface
    
    surface_density_scale_radius は、Σ_gas(R) ∝ I_emission(R) と仮定すればI(R)のscale radiusと同じ値を使える
    radiusの単位は、scale_radiusと揃っていれば任意
    """

    if vertical_model == "constant_scale_height":
        pressure_slope = 1.0 / surface_density_scale_radius + 2.0 / velocity_dispersion_scale_radius
    elif vertical_model == "self_gravitating_hydrostatic":
        pressure_slope = 2.0 / surface_density_scale_radius # https://ui.adsabs.harvard.edu/abs/2010ApJ...725.2324B/abstract Eq.11
    else:
        raise ValueError("vertical_model must be 'constant_scale_height' or 'self_gravitating_hydrostatic'.")

    v_rot2 = circular_velocity_kms**2 - radial_velocity_dispersion_kms**2 * radius * pressure_slope

    # pressure supportが重力より大きくなったらエラー
    if v_rot2.min() < 0.0:
        index = int(np.flatnonzero(v_rot2 < 0.0)[0])
        raise UnphysicalPressureSupportError(f"pressure support gives v_rot^2 < 0 at radius index {index}.")

    return np.sqrt(v_rot2)
