from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


FIDUCIAL_COSMOLOGY = FlatLambdaCDM(
    H0 = 70,
    Om0 = 0.3,
    Tcmb0 = 2.725,
)

def kpc_per_arcsec(
    redshift: float,
    cosmology=FIDUCIAL_COSMOLOGY,
) -> float:
    return cosmology.kpc_proper_per_arcmin(redshift).to_value(
        u.kpc / u.arcsec
    )

def critical_density(
    redshift: float,
    cosmology=FIDUCIAL_COSMOLOGY,
) -> float:
    return cosmology.critical_density(redshift).to_value(
        u.Msun / u.kpc**3
)