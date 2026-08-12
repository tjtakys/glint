"""
Forward modelのパラメータ名と順序をここにハードコードして一元管理する。
TOMLで指定したfree parameterとサンプラーの数値配列を対応付け、固定値との合成、prior変換、名前付きアクセスを扱えるようにする。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, NamedTuple, Sequence
import numpy as np


MASS_PARAMETER_NAMES = (
    "log10_m_baryon",
    "disk_scale_radius_kpc",
    "log10_m200",
    "bulge_to_total",
    "bulge_effective_radius_kpc",
)
SOURCE_PARAMETER_NAMES = (
    "source_x_arcsec",
    "source_y_arcsec",
    "flux_normalization",
    "inclination_deg",
    "position_angle_deg",
    "surface_brightness_scale_arcsec",
    "velocity_dispersion_center_kms",
    "velocity_dispersion_scale_arcsec",
    "systemic_velocity_kms",
)
LENS_PARAMETER_NAMES = (
    "lens_x_arcsec",
    "lens_y_arcsec",
    "einstein_radius_arcsec",
    "lens_axis_ratio",
    "lens_position_angle_deg",
    "log10_external_shear",
    "external_shear_position_angle_deg",
    "external_convergence",
)
ALL_PARAMETER_NAMES = (
    MASS_PARAMETER_NAMES + SOURCE_PARAMETER_NAMES + LENS_PARAMETER_NAMES
)

PARAMETER_INDEX = {
    name: index for index, name in enumerate(ALL_PARAMETER_NAMES)
}


class ModelParameters(NamedTuple):
    """
    名前またはindexでアクセスできる Forward modelで使用する全パラメータ
    ALL_PARAMETER_NAMESの順序で保持する（ndarrayに変換してサンプラーに渡すので順番変えるのはNG）
    サンプリング中にインスタンス生成するが、時間はほぼ無視できる（<<0.1%）
    初期設定時にだけ呼ばれ、サンプリング中には新たに作ったりしない
    """
    # mass parameters
    log10_m_baryon: float
    disk_scale_radius_kpc: float
    log10_m200: float
    bulge_to_total: float
    bulge_effective_radius_kpc: float

    # source parameters
    source_x_arcsec: float
    source_y_arcsec: float
    flux_normalization: float
    inclination_deg: float
    position_angle_deg: float
    surface_brightness_scale_arcsec: float
    velocity_dispersion_center_kms: float
    velocity_dispersion_scale_arcsec: float
    systemic_velocity_kms: float

    # lens parameters
    lens_x_arcsec: float
    lens_y_arcsec: float
    einstein_radius_arcsec: float
    lens_axis_ratio: float
    lens_position_angle_deg: float
    log10_external_shear: float
    external_shear_position_angle_deg: float
    external_convergence: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> ModelParameters:
        supplied = set(values)
        expected = set(ALL_PARAMETER_NAMES)
        missing = expected - supplied
        unknown = supplied - expected
        if missing or unknown:
            raise ValueError(
                f"Invalid model parameters: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(*(float(values[name]) for name in ALL_PARAMETER_NAMES))

    def as_array(self) -> np.ndarray:
        # convert to a numpy array in the order of ALL_PARAMETER_NAMES
        return np.fromiter(self, dtype=float, count=len(ALL_PARAMETER_NAMES))


# 最初の一回だけ呼び出し、サンプリングでは作成しない
@dataclass(frozen=True, slots=True)
class ParameterLayout:
    """
    TOMLで指定した固定・自由パラメータとprior範囲を反映する。
    サンプリング中に毎回呼ばれるのは decode/encode/prior_transform
    """

    base_values: np.ndarray  # 初期値 or 固定値
    free_names: tuple[str, ...]
    free_indices: np.ndarray
    prior_lower: np.ndarray
    prior_span: np.ndarray

    @classmethod
    def compile(
        cls,
        baseline: ModelParameters,
        free_names: Sequence[str],
        bounds: Mapping[str, Mapping[str, float]],
        *,
        no_bulge: bool = False,
    ) -> ParameterLayout:

        # Check free params
        names = tuple(free_names)
        if len(set(names)) != len(names):
            raise ValueError("free parameter names contain duplicates")
        unknown = set(names) - set(ALL_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"Unsupported free parameters: {sorted(unknown)}")
        if no_bulge and "bulge_to_total" in names:
            raise ValueError("bulge_to_total cannot be free in a no-bulge model")

        base_values = baseline.as_array()
        if no_bulge:
            base_values[PARAMETER_INDEX["bulge_to_total"]] = 0.0
        free_indices = np.fromiter(
            (PARAMETER_INDEX[name] for name in names),
            dtype=np.intp,
            count=len(names),
        )

        # Check bounds
        missing_bounds = set(names) - set(bounds)
        if missing_bounds:
            raise ValueError(f"Missing bounds: {sorted(missing_bounds)}")
        prior_lower = np.fromiter(
            (float(bounds[name]["lower"]) for name in names),
            dtype=float,
            count=len(names),
        )
        prior_upper = np.fromiter(
            (float(bounds[name]["upper"]) for name in names),
            dtype=float,
            count=len(names),
        )
        if not np.all(np.isfinite(prior_lower)) or not np.all(np.isfinite(prior_upper)):
            raise ValueError("parameter bounds must be finite")
        if np.any(prior_lower >= prior_upper):
            raise ValueError("every lower bound must be smaller than its upper bound")
        prior_span = prior_upper - prior_lower

        # Make arrays read-only to prevent accidental modification
        for array in (base_values, free_indices, prior_lower, prior_span):
            array.setflags(write=False)
        return cls(base_values, names, free_indices, prior_lower, prior_span)

    @property
    def ndim(self) -> int:
        return len(self.free_names)

    def decode(self, theta: np.ndarray) -> ModelParameters:
        # サンプラーから返されるthetaを、固定パラメータも含めた全パラメータに展開する。サンプリング中に毎回呼ばれる。
        theta_array = np.asarray(theta, dtype=float)
        if theta_array.shape != (self.ndim,):
            raise ValueError(f"theta must have shape ({self.ndim},); got {theta_array.shape}")
        values = self.base_values.copy()
        values[self.free_indices] = theta_array
        return ModelParameters(*values.tolist())

    def encode(self, parameters: ModelParameters) -> np.ndarray:
        return parameters.as_array()[self.free_indices]

    def prior_transform(self, unit_cube: np.ndarray) -> np.ndarray:
        # Dynesty用のprior変換。log対応はまだ。
        unit_array = np.asarray(unit_cube, dtype=float)
        if unit_array.shape != (self.ndim,):
            raise ValueError(
                f"unit_cube must have shape ({self.ndim},); got {unit_array.shape}"
            )
        return self.prior_lower + unit_array * self.prior_span
