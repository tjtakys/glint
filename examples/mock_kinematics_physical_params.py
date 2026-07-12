"""Physical parameter grid for the image-plane kinematic mock experiments.

This file contains only experiment definitions.  The derived Courteau
parameters are written by ``fit_mock_courteau_params.py`` to
``mock_kinematics_courteau_fits.json`` so that the recovery notebook can load
the fitted values without repeating the physical-model fit.
"""

from __future__ import annotations

from itertools import product


# Cosmology used for G09.
SOURCE_REDSHIFT = 6.024
H0_KM_S_MPC = 70.0
OMEGA_M = 0.3

# Keep the total baryonic mass fixed while redistributing it between disk and
# bulge.  Every case includes the same modest NFW halo.
TOTAL_BARYONIC_MASS_MSUN = 1.0e11
DISK_SCALE_RADIUS_KPC = 1.00
BULGE_PROFILE = "Hernquist"
BULGE_TO_TOTAL_VALUES = (0.10, 0.25, 0.50)
BULGE_EFFECTIVE_RADII_KPC = (0.30, 0.60)
HALO_MASS_200_MSUN = 3.0e11
HALO_CONCENTRATION_200 = 4.0

# Match the source-plane radial extent used by the current kinematic notebook.
# The innermost fitted radius is non-zero to avoid giving the mathematical
# origin disproportionate influence over the fit.
FIT_RADIUS_MIN_ARCSEC = 0.02
FIT_RADIUS_MAX_ARCSEC = 0.80
FIT_RADIUS_SAMPLES = 600

# Broad bounds used to test the flexibility of the Courteau parameterization,
# rather than the narrower priors currently used for inference.
COURTEAU_INITIAL = {
    "v_c_kms": 300.0,
    "r_turn_arcsec": 0.10,
    "beta": 0.0,
    "gamma": 2.5,
}
COURTEAU_FIT_BOUNDS = {
    "v_c_kms": (20.0, 800.0),
    "r_turn_arcsec": (0.001, 2.0),
    "beta": (-3.0, 0.999),
    "gamma": (0.10, 20.0),
}

# Current bounds in params_kinematics.py.  These are not imposed on the
# physical-to-Courteau fit; they are recorded to diagnose whether a fitted
# curve can be injected and recovered with the existing notebook priors.
CURRENT_INFERENCE_BOUNDS = {
    "v_c_kms": (200.0, 500.0),
    "r_turn_arcsec": (0.005, 0.50),
    "beta": (0.50, 1.00),
    "gamma": (1.00, 4.50),
}


def case_name(
    bulge_to_total: float,
    bulge_effective_radius_kpc: float,
) -> str:
    """Return a stable, filesystem-safe name for one physical case."""
    bt_label = f"bt{int(round(100 * bulge_to_total)):02d}"
    re_label = f"re{int(round(10 * bulge_effective_radius_kpc)):02d}"
    return f"{bt_label}_{re_label}_dm"


def physical_cases() -> tuple[dict[str, float | str | bool], ...]:
    """Return the six disk-plus-bulge cases, all with the same NFW halo."""
    cases = []
    for bulge_to_total, bulge_effective_radius_kpc in product(
        BULGE_TO_TOTAL_VALUES,
        BULGE_EFFECTIVE_RADII_KPC,
    ):
        bulge_mass_msun = TOTAL_BARYONIC_MASS_MSUN * bulge_to_total
        disk_mass_msun = TOTAL_BARYONIC_MASS_MSUN - bulge_mass_msun
        cases.append(
            {
                "name": case_name(bulge_to_total, bulge_effective_radius_kpc),
                "bulge_to_total": bulge_to_total,
                "bulge_effective_radius_kpc": bulge_effective_radius_kpc,
                "bulge_mass_msun": bulge_mass_msun,
                "disk_mass_msun": disk_mass_msun,
                "total_baryonic_mass_msun": TOTAL_BARYONIC_MASS_MSUN,
                "disk_scale_radius_kpc": DISK_SCALE_RADIUS_KPC,
                "bulge_profile": BULGE_PROFILE,
                "has_dark_matter": True,
                "halo_mass_200_msun": HALO_MASS_200_MSUN,
                "halo_concentration_200": HALO_CONCENTRATION_200,
            }
        )
    return tuple(cases)


def validate_physical_cases() -> None:
    """Fail early if the experiment grid is accidentally made inconsistent."""
    cases = physical_cases()
    if len(cases) != 6:
        raise ValueError(f"Expected six physical cases, found {len(cases)}")
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Physical case names must be unique")
    for case in cases:
        component_sum = case["disk_mass_msun"] + case["bulge_mass_msun"]
        if component_sum != TOTAL_BARYONIC_MASS_MSUN:
            raise ValueError(f"Mass components do not sum to the total in {case['name']}")


validate_physical_cases()
