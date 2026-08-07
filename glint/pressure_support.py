"""Pressure support for exponential gas profiles."""

from __future__ import annotations
from typing import Literal
import numpy as np


VerticalModel = Literal["constant_scale_height", "self_gravitating_hydrostatic"]


def pressure_supported_rotation_velocity(
    radius: np.ndarray,
    circular_velocity_kms: np.ndarray,
    radial_velocity_dispersion_kms: np.ndarray,
    surface_density_scale_radius: float,
    velocity_dispersion_scale_radius: float | None = None,
    *,
    vertical_model: VerticalModel = "constant_scale_height",
) -> np.ndarray:
    
    """指数関数profileにpressure supportを適用した回転速度を計算
    surface_density(R) ∝ exp(-R / R_surface)
    sigma_R(R)         ∝ exp(-R / R_sigma)

    ↑どちらも指数関数を仮定しているので、解析的に計算可能
    constant_scale_height:
        v_rot**2 = v_circ**2
                    - sigma_R**2 * R * (1 / R_surface + 2 / R_sigma)
    self_gravitating_hydrostatic:
        v_rot**2 = v_circ**2 - 2 * sigma_R**2 * R / R_surface
    """

    if vertical_model == "constant_scale_height":
        pressure_slope = (
            1.0 / surface_density_scale_radius
            + 2.0 / velocity_dispersion_scale_radius
        )
    elif vertical_model == "self_gravitating_hydrostatic":
        pressure_slope = 2.0 / surface_density_scale_radius
    else:
        raise ValueError(
            "vertical_model must be 'constant_scale_height' or "
            "'self_gravitating_hydrostatic'."
        )

    rotation_velocity2 = (
        circular_velocity_kms**2
        - radial_velocity_dispersion_kms**2 * radius * pressure_slope
    )

    # 有効な個々のparameterでも、組合せによって非物理的になり得る。
    if rotation_velocity2.min() < 0.0:
        index = int(np.flatnonzero(rotation_velocity2 < 0.0)[0])
        raise ValueError(
            "pressure support gives v_rot^2 < 0 at "
            f"radius index {index}."
        )

    return np.sqrt(rotation_velocity2)
