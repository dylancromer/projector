import numpy as np
import scipy.integrate as integrate
import projector.mathutils as mathutils


N_SAMPLES = 60


def _first_term_integrand_func(xs, radii, density_func):
    return 4 * density_func(xs) * xs**2 / mathutils.atleast_kd(radii, xs.ndim)**2


def _second_term_integrand_func_theta(thetas, radii, density_func):
    radii = radii[:, None]
    thetas = thetas[None, :]
    return 4 * radii * density_func(radii/np.cos(thetas)) / (4*np.sin(thetas) + 3 - np.cos(2*thetas))


def _second_term_integrand_func_x(xs, radii, density_func):
    rs = mathutils.atleast_kd(radii, xs.ndim)
    return 2*xs * (2*xs + (rs**2 - 2*xs**2)/(np.sqrt(xs**2 - rs**2))) * density_func(xs)


def esd(radii, density_func):
    xs = np.stack((np.linspace(1e-6, radius, N_SAMPLES) for radius in radii))
    dxs = np.gradient(xs, axis=-1).T
    first_term_integrand = _first_term_integrand_func(xs, radii, density_func)
    first_term = mathutils._trapz(first_term_integrand, axis=-1, dx=dxs)

    thetas = np.linspace(0, np.pi/2, N_SAMPLES)
    dthetas = np.gradient(thetas, axis=-1).T
    second_term_integrand = _second_term_integrand_func_theta(thetas, radii, density_func)
    second_term = mathutils._trapz(second_term_integrand, axis=-1, dx=dthetas)

    return first_term - second_term
