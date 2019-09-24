import numpy as np
import scipy.integrate as integrate
import cubature


def _first_term_integrand_func(xs, radii, density_func):
    return 4 * density_func(xs) * xs**2 / radii**2


def _second_term_integrand_func_theta(thetas, radii, density_func):
    return 4 * radii * density_func(radii/np.cos(thetas)) / (4*np.sin(thetas) + 3 - np.cos(2*thetas))


def _second_term_integrand_func_x(xs, radii, density_func):
    return 2*xs * (2*xs + (radii**2 - 2*xs**2)/(np.sqrt(xs**2 - radii**2))) * density_func(xs)


def esd(radii, density_func):
    first_term = np.array(
        [integrate.quad(_first_term_integrand_func, 0, r, args=(r, density_func))[0] for r in radii]
    )

    second_term = np.array(
        [integrate.quad(_second_term_integrand_func_theta, 0, np.pi/2, args=(r, density_func))[0] for r in radii]
    )

    return first_term - second_term
