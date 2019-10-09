import numpy as np
import scipy.integrate as integrate
import projector.mathutils as mathutils


def _first_term_integrand_func(xs, radii, density_func, radius_axis):
    rhos = density_func(xs)
    postfactor = xs**2 / mathutils.atleast_kd(radii, xs.ndim)**2

    if radius_axis == -1:
        radius_axis = rhos.ndim

    x_axes = tuple(range(radius_axis, radius_axis+xs.ndim))
    assert False, tuple(rhos.shape[i] for i in x_axes)

    k_before = len(rhos.shape[:radius_axis]) - postfactor.ndim
    k_after = len(rhos.shape[radius_axis:])
    postfactor = mathutils.extend_shape(postfactor, k_before, k_after)
    return 4 * rhos * postfactor


def _second_term_integrand_func(thetas, radii, density_func):
    radii = radii[:, None]
    thetas = thetas[None, :]
    return 4 * radii * density_func(radii/np.cos(thetas)) / (4*np.sin(thetas) + 3 - np.cos(2*thetas))


def esd(radii, density_func, num_points=200, radius_axis=-1):
    xs = np.stack(tuple(np.linspace(1e-6, radius, num_points) for radius in radii)).T
    first_term_integrand = _first_term_integrand_func(xs, radii, density_func, radius_axis=radius_axis)

    dxs = mathutils.atleast_kd(np.gradient(xs, axis=radius_axis), first_term_integrand.ndim, insertion_axis=0)
    first_term = mathutils.trapz_(first_term_integrand, axis=radius_axis, dx=dxs)

    thetas = np.linspace(0, np.pi/2, num_points)
    dthetas = np.gradient(thetas, axis=-1)
    second_term_integrand = _second_term_integrand_func(thetas, radii, density_func)
    second_term = mathutils.trapz_(second_term_integrand, axis=-1, dx=dthetas)

    return first_term - second_term
