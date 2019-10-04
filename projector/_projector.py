import numpy as np
import scipy.integrate as integrate
import projector.mathutils as mathutils


def _first_term_integrand_func(xs, radii, density_func):
    rhos = density_func(xs)
    postfactor = xs**2 / mathutils.atleast_kd(radii, xs.ndim)**2
    postfactor = mathutils.atleast_kd(postfactor, rhos.ndim, append_dims=False)
    return 4 * rhos * postfactor


def _second_term_integrand_func_theta(thetas, radii, density_func):
    radii = radii[:, None]
    thetas = thetas[None, :]
    return 4 * radii * density_func(radii/np.cos(thetas)) / (4*np.sin(thetas) + 3 - np.cos(2*thetas))


def _second_term_integrand_func_x(xs, radii, density_func):
    rs = mathutils.atleast_kd(radii, xs.ndim)
    return 2*xs * (2*xs + (rs**2 - 2*xs**2)/(np.sqrt(xs**2 - rs**2))) * density_func(xs)


def esd(radii, density_func, num_points=100):
    xs = np.stack((np.linspace(1e-6, radius, num_points) for radius in radii))
    first_term_integrand = _first_term_integrand_func(xs, radii, density_func)

    dxs = mathutils.atleast_kd(np.gradient(xs, axis=-1), first_term_integrand.ndim, append_dims=False)
    first_term = mathutils.trapz_(first_term_integrand, axis=-1, dx=dxs)

    thetas = np.linspace(0, np.pi/2, num_points)
    dthetas = np.gradient(thetas, axis=-1).T
    second_term_integrand = _second_term_integrand_func_theta(thetas, radii, density_func)
    second_term = mathutils.trapz_(second_term_integrand, axis=-1, dx=dthetas)

    return first_term - second_term
