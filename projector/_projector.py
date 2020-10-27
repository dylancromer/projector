from dataclasses import dataclass
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


@dataclass
class SurfaceDensity:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    density_axis: object

    def _pad_radii_and_thetas_for_argument(self, thetas, radii):
        radii_ = radii.reshape(radii.shape + thetas.ndim*(1,))
        thetas_ = thetas.reshape(radii.ndim*(1,) + thetas.shape)
        if self.radial_axis_to_broadcast is not None:
            radii_ = np.moveaxis(radii_, self.radial_axis_to_broadcast, -1)
            thetas_ = np.moveaxis(thetas_, self.radial_axis_to_broadcast, -1)
        return thetas_, radii_

    def _pad_radii_and_thetas_for_integrand(self, thetas, radii, rhos):
        radii_ = radii.reshape(radii.shape + thetas.ndim*(1,))
        thetas_ = thetas.reshape(radii.ndim*(1,) + thetas.shape)
        radii__ = mathutils.atleast_kd(radii_, rhos.ndim)
        thetas__ = mathutils.atleast_kd(thetas_, rhos.ndim)
        if self.radial_axis_to_broadcast is not None:
            radii__ = np.moveaxis(radii__, self.radial_axis_to_broadcast, self.density_axis)
            thetas__ = np.moveaxis(thetas__, self.radial_axis_to_broadcast, self.density_axis)
        return thetas__, radii__

    def _sd_integrand_func(self, thetas):
        thetas_, radii_ = self._pad_radii_and_thetas_for_argument(thetas, self.radii)
        density_arg = radii_/np.cos(thetas_)
        rhos = self.density_func(density_arg)
        thetas__, radii__ = self._pad_radii_and_thetas_for_integrand(thetas, self.radii, rhos)
        return 2 * radii__ * rhos/(np.cos(thetas__)**2)

    def sd(self):
        thetas = np.linspace(0, np.pi/2, self.num_points)
        dthetas = np.gradient(thetas, axis=0)
        integrand = self._sd_integrand_func(thetas)
        return mathutils.trapz_(integrand, axis=1, dx=dthetas)

    @classmethod
    def calculate(cls, radii, density_func, num_points=120, radial_axis_to_broadcast=None, density_axis=-1):
        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
            density_axis=density_axis,
        ).sd()


@dataclass
class SurfaceDensity2:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object

    def _pad_radii_and_ells_for_argument(self, ells, radii):
        radii_ = mathutils.atleast_kd(radii, radii.ndim+ells.ndim)
        ells_ = mathutils.atleast_kd(ells, radii.ndim+ells.ndim, append_dims=False)
        if self.radial_axis_to_broadcast is not None:
            radii_ = np.moveaxis(radii_, self.radial_axis_to_broadcast, radii_.ndim-1)
            ells_ = np.moveaxis(ells_, self.radial_axis_to_broadcast, ells_.ndim-1)
        return ells_, radii_

    def _sd_alt_integrand_func(self, ells):
        ells_, radii_ = self._pad_radii_and_ells_for_argument(ells, self.radii)
        density_arg = np.sqrt(radii_**2 + ells_**2)
        rhos = self.density_func(density_arg)
        return 2*rhos

    def sd(self):
        ells = np.geomspace(MIN_INTEGRATION_RADIUS, MAX_INTEGRATION_RADIUS, self.num_points)
        d_ells = np.gradient(ells, axis=0)
        integrand = self._sd_alt_integrand_func(ells)
        return mathutils.trapz_(integrand, axis=1, dx=d_ells)

    @classmethod
    def calculate(cls, radii, density_func, num_points=120, radial_axis_to_broadcast=None, density_axis=None):
        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
        ).sd()


@dataclass
class ExcessSurfaceDensity:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    density_axis: object

    def _esd_first_term_integrand_func(self, xs):
        rhos = self.density_func(xs)
        radii_ = self.radii
        postfactor = xs**2 / mathutils.atleast_kd(self.radii, xs.ndim, append_dims=False)**2
        postfactor = mathutils.atleast_kd(postfactor, rhos.ndim)
        if self.radial_axis_to_broadcast is not None:
            postfactor = np.moveaxis(postfactor, self.radial_axis_to_broadcast+1, self.density_axis)
        return 4 * rhos * postfactor

    def _esd_second_term_integrand_func(self, thetas):
        thetas_ = mathutils.atleast_kd(thetas, self.radii.ndim+1)
        density_arg = self.radii[None, ...]/np.cos(thetas_)
        rhos = self.density_func(density_arg)
        radii = mathutils.atleast_kd(self.radii[None, ...], rhos.ndim)
        if self.radial_axis_to_broadcast is not None:
            radii = np.moveaxis(radii, self.radial_axis_to_broadcast+1, self.density_axis)
        thetas__ = mathutils.atleast_kd(thetas_, rhos.ndim)
        return 4 * radii * rhos / (4*np.sin(thetas__) + 3 - np.cos(2*thetas__))

    def esd(self):
        xs = np.linspace(MIN_INTEGRATION_RADIUS, self.radii, self.num_points)
        first_term_integrand = self._esd_first_term_integrand_func(xs)

        dxs = mathutils.atleast_kd(np.gradient(xs, axis=0), first_term_integrand.ndim)
        if self.radial_axis_to_broadcast is not None:
            dxs = np.moveaxis(dxs, self.radial_axis_to_broadcast+1, self.density_axis)
        first_term = mathutils.trapz_(first_term_integrand, axis=0, dx=dxs)

        thetas = np.linspace(0, np.pi/2, self.num_points)
        dthetas = np.gradient(thetas, axis=0)
        second_term_integrand = self._esd_second_term_integrand_func(thetas)
        second_term = mathutils.trapz_(second_term_integrand, axis=0, dx=dthetas)

        return first_term - second_term

    @classmethod
    def calculate(cls, radii, density_func, num_points=120, radial_axis_to_broadcast=None, density_axis=None):
        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
            density_axis=density_axis,
        ).esd()


@dataclass
class QuadExcessSurfaceDensity:
    radii: np.ndarray
    density_func: object
    num_points: int
    radial_axis_to_broadcast: object
    density_axis: object

    def _esd_first_term_integrand_func(self, xs, radius):
        rhos = self.density_func(xs)
        postfactor = xs**2 / mathutils.atleast_kd(radius, xs.ndim, append_dims=False)**2
        postfactor = mathutils.atleast_kd(postfactor, rhos.ndim)
        return 4 * rhos * postfactor

    def _esd_second_term_integrand_func(self, thetas):
        thetas_ = mathutils.atleast_kd(thetas, self.radii.ndim+1)
        density_arg = self.radii[None, ...]/np.cos(thetas_)
        rhos = self.density_func(density_arg)
        radii = mathutils.atleast_kd(self.radii[None, ...], rhos.ndim)
        thetas = mathutils.atleast_kd(thetas_, rhos.ndim)
        return 4 * radii * rhos / (4*np.sin(thetas) + 3 - np.cos(2*thetas))

    def _integrate_esd_first_term_quad(self):
        rflats = self.radii.flatten()
        first_term_integral = np.array(
            [integrate.quad_vec(
                lambda x: self._esd_first_term_integrand_func(np.array([x]), rflats[i:i+1]),
                MIN_INTEGRATION_RADIUS,
                rflats[i],
            ) for i in range(self.radii.size)],
            dtype=object,
        )
        _check_errors_ok(first_term_integral[:, 1].astype(float))
        return np.concatenate(first_term_integral[:, 0]).astype(float)

    def _integrate_esd_second_term_quad(self):
        second_term, second_term_errors = integrate.quad_vec(
            lambda theta: self._esd_second_term_integrand_func(np.array([theta])),
            0,
            np.pi/2,
        )
        _check_errors_ok(second_term_errors)
        return second_term

    def esd(self):
        dens_shape = self.density_func(self.radii).shape
        first_term = self._integrate_esd_first_term_quad().reshape(dens_shape)
        second_term = self._integrate_esd_second_term_quad().reshape(dens_shape)
        return first_term - second_term

    @classmethod
    def calculate(cls, radii, density_func, num_points=120, radial_axis_to_broadcast=None, density_axis=None):
        return cls(
            radii=radii,
            density_func=density_func,
            num_points=num_points,
            radial_axis_to_broadcast=radial_axis_to_broadcast,
            density_axis=density_axis,
        ).esd()
