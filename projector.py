import numpy as np
import cubature


def _first_term_integrand_func(xs, radii, density_func):
    return 4 * density_func(xs) * xs**2 / radii**2


def _second_term_integrand_func(thetas, radii, density_func):
    return 4 * radii * density_func(radii/np.cos(thetas)) / (4*np.sin(thetas) + 3 - np.cos(2*thetas))


def _second_term_integrand_func(xs, radii, density_func):
    return 2*xs * (2*xs + (radii**2 - 2*xs**2)/(np.sqrt(xs**2 - radii**2))) * density_func(xs)


def esd(radii, density_func):
    first_term = cubature.cubature(
        func=_first_term_integrand_func,
        ndim=radii.size,
        fdim=radii.size,
        xmin=np.zeros(radii.size),
        xmax=radii,
        args=(radii, density_func),
        adaptive='p',
        relerr=1e-3,
        vectorized=True,
    )

    second_term = cubature.cubature(
        func=_second_term_integrand_func,
        ndim=radii.size,
        fdim=radii.size,
        xmin=radii,
        xmax=1e2*radii,
        args=(radii, density_func),
        adaptive='p',
        vectorized=True,
        relerr=1e-3,
    )

    assert False, second_term

    return first_term - second_term
