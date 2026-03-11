def sersic(radius, M_bulge, R_eff, n):
    """Calculate the Sersic profile for a given bulge mass, effective radius, and Sersic index."""
    from scipy.special import gamma

    # Calculate the Sersic constant b_n
    b_n = 2 * n - 1/3 + 0.00987654321 / n + 0.001802861062 / n**2 # For n > 0.36 (Ciotti & Bertin 1999)
    p = 1.0 - 0.6097 / n + 0.05563 / n**2  # For 0.6 < n < 10 (Terzic & Graham 2005)

    # velocity contribution
    from astropy.constants import G
    from astropy import units as u
    G = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun)
    vel_sq 