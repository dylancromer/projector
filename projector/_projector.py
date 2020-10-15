import numpy as np
import scipy.integrate as integrate
import projector.mathutils as mathutils


MIN_INTEGRATION_RADIUS = 1e-6
MAX_INTEGRATION_RADIUS = 1e4
MAX_ERROR_TOLERANCE = 10


class LargeQuadratureErrorsException(Exception):
    pass


def _check_errors_ok(error):
    try:
        max_error = error.max()
    except AttributeError:
        max_error = error
    if max_error > MAX_ERROR_TOLERANCE:
        raise LargeQuadratureErrorsException(
            f'Maximum quadrature error ({round(max_error, 2)}) is very large and indicates a problem',
        )


def _sd_integrand_func(thetas, radii, density_func, radial_axis_to_broadcast):
    radii_ = mathutils.atleast_kd(radii, radii.ndim+thetas.ndim)
    thetas_ = mathutils.atleast_kd(thetas, radii.ndim+thetas.ndim, append_dims=False)
    if radial_axis_to_broadcast is not None:
        radii_ = np.moveaxis(radii_, radial_axis_to_broadcast, radii_.ndim-1)
        thetas_ = np.moveaxis(thetas_, radial_axis_to_broadcast, thetas_.ndim-1)
    density_arg = radii_/np.cos(thetas_)
    rhos = density_func(density_arg)
    radii__ = mathutils.atleast_kd(radii_, rhos.ndim)
    thetas__ = mathutils.atleast_kd(thetas_, rhos.ndim)
    return 2 * radii__ * rhos/(np.cos(thetas__)**2)


def sd(radii, density_func, num_points=120, radial_axis_to_broadcast=None):
    thetas = np.linspace(0, np.pi/2, num_points)
    dthetas = np.gradient(thetas, axis=0)
    integrand = _sd_integrand_func(thetas, radii, density_func, radial_axis_to_broadcast)
    return mathutils.trapz_(integrand, axis=1, dx=dthetas)


def _sd_alt_integrand_func(ells, radii, density_func, radial_axis_to_broadcast):
    radii_ = mathutils.atleast_kd(radii, radii.ndim+ells.ndim)
    ells_ = mathutils.atleast_kd(ells, radii.ndim+ells.ndim, append_dims=False)
    if radial_axis_to_broadcast is not None:
        radii_ = np.moveaxis(radii_, radial_axis_to_broadcast, radii_.ndim-1)
        ells_ = np.moveaxis(ells_, radial_axis_to_broadcast, ells_.ndim-1)
    density_arg = np.sqrt(radii_**2 + ells_**2)
    rhos = density_func(density_arg)
    radii__ = mathutils.atleast_kd(radii_, rhos.ndim)
    ells__ = mathutils.atleast_kd(ells_, rhos.ndim)
    return 2*rhos


def sd_alt(radii, density_func, num_points=120, radial_axis_to_broadcast=None):
    ells = np.geomspace(MIN_INTEGRATION_RADIUS, MAX_INTEGRATION_RADIUS, num_points)
    d_ells = np.gradient(ells, axis=0)
    integrand = _sd_alt_integrand_func(ells, radii, density_func, radial_axis_to_broadcast)
    return mathutils.trapz_(integrand, axis=1, dx=d_ells)


def _esd_first_term_integrand_func(xs, radii, density_func):
    rhos = density_func(xs)
    postfactor = xs**2 / mathutils.atleast_kd(radii, xs.ndim)**2
    postfactor = mathutils.atleast_kd(postfactor, rhos.ndim)
    return 4 * rhos * postfactor


def _esd_second_term_integrand_func(thetas, radii, density_func):
    density_arg = radii[:, None]/np.cos(thetas[None, :])
    rhos = density_func(density_arg)
    radii = mathutils.atleast_kd(radii[:, None], rhos.ndim)
    thetas = mathutils.atleast_kd(thetas[None, :], rhos.ndim)
    return 4 * radii * rhos / (4*np.sin(thetas) + 3 - np.cos(2*thetas))


def esd(radii, density_func, num_points=120):
    xs = np.stack(tuple(np.linspace(MIN_INTEGRATION_RADIUS, radius, num_points) for radius in radii))
    first_term_integrand = _esd_first_term_integrand_func(xs, radii, density_func)

    dxs = mathutils.atleast_kd(np.gradient(xs, axis=1), first_term_integrand.ndim)
    first_term = mathutils.trapz_(first_term_integrand, axis=1, dx=dxs)

    thetas = np.linspace(0, np.pi/2, num_points)
    dthetas = np.gradient(thetas, axis=0)
    second_term_integrand = _esd_second_term_integrand_func(thetas, radii, density_func)
    second_term = mathutils.trapz_(second_term_integrand, axis=1, dx=dthetas)

    return first_term - second_term


def _integrate_esd_first_term_quad(radii, density_func):
    rflats = radii.flatten()
    first_term_integral = np.array(
        [integrate.quad_vec(
            lambda x: _esd_first_term_integrand_func(np.array([x]), rflats[i:i+1], density_func),
            MIN_INTEGRATION_RADIUS,
            rflats[i],
        ) for i in range(radii.size)],
        dtype=object,
    )
    _check_errors_ok(first_term_integral[:, 1].astype(float))
    return np.concatenate(first_term_integral[:, 0]).astype(float)


def _integrate_esd_second_term_quad(radii, density_func):
    second_term, second_term_errors = integrate.quad_vec(
        lambda theta: _esd_second_term_integrand_func(np.array([theta]), radii, density_func),
        0,
        np.pi/2,
    )
    _check_errors_ok(second_term_errors)
    return second_term


def esd_quad(radii, density_func):
    dens_shape = density_func(radii).shape
    first_term = _integrate_esd_first_term_quad(radii, density_func).reshape(dens_shape)
    second_term = _integrate_esd_second_term_quad(radii, density_func).reshape(dens_shape)
    return first_term - second_term


