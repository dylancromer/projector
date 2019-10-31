import numpy as np
import scipy.integrate as integrate
import projector.mathutils as mathutils


def _first_term_integrand_func(xs, radii, density_func):
    rhos = density_func(xs)
    postfactor = xs**2 / mathutils.atleast_kd(radii, xs.ndim)**2
    postfactor = mathutils.atleast_kd(postfactor, rhos.ndim)
    return 4 * rhos * postfactor


def _second_term_integrand_func(thetas, radii, density_func):
    density_arg = radii[:, None]/np.cos(thetas[None, :])
    rhos = density_func(density_arg)
    radii = mathutils.atleast_kd(radii[:, None], rhos.ndim)
    thetas = mathutils.atleast_kd(thetas[None, :], rhos.ndim)
    return 4 * radii * rhos / (4*np.sin(thetas) + 3 - np.cos(2*thetas))


def esd(radii, density_func, num_points=120):
    xs = np.stack(tuple(np.linspace(1e-6, radius, num_points) for radius in radii))
    first_term_integrand = _first_term_integrand_func(xs, radii, density_func)

    dxs = mathutils.atleast_kd(np.gradient(xs, axis=1), first_term_integrand.ndim)
    first_term = mathutils.trapz_(first_term_integrand, axis=1, dx=dxs)

    thetas = np.linspace(0, np.pi/2, num_points)
    dthetas = np.gradient(thetas, axis=0)
    second_term_integrand = _second_term_integrand_func(thetas, radii, density_func)
    second_term = mathutils.trapz_(second_term_integrand, axis=1, dx=dthetas)

    return first_term - second_term
