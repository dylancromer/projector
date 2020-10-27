import pytest
import numpy as np
import projector


def describe_SurfaceDensity():

    def it_projects_an_inverse_square_law_correctly():
        radii = np.geomspace(0.1, 10, 30)
        rho_func = lambda r: 1/r**2
        sds = projector.SurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(sds, np.pi/radii, rtol=1e-4)

    def it_can_handle_larger_shapes():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: projector.mathutils.atleast_kd(1/r, r.ndim + 3) * np.random.random_sample(r.shape + (3, 4, 5))
        sds = projector.SurfaceDensity.calculate(radii, rho_func)

        assert not np.any(np.isnan(sds))

    def it_can_allow_radii_that_are_functions_of_other_parameters():
        def rho_func(r, z):
            z_ = projector.mathutils.atleast_kd(z, r.ndim, append_dims=False)
            return (z_/r**2)[..., None]

        zs = np.linspace(1, 2, 3)
        rs = np.array([
            np.linspace(i, i+1, 10) for i in range(1, zs.size+1)
        ]).T
        sds = projector.SurfaceDensity.calculate(rs, lambda r: rho_func(r, zs), radial_axis_to_broadcast=1, density_axis=-2)

        zs = projector.mathutils.atleast_kd(zs, rs.ndim, append_dims=False)

        assert np.allclose(sds, (zs * np.pi/rs)[..., None], rtol=1e-4)


def describe_SurfaceDensity2():

    def it_projects_an_inverse_square_law_correctly():
        radii = np.geomspace(0.1, 10, 30)
        rho_func = lambda r: 1/r**2
        sds = projector.SurfaceDensity2.calculate(radii, rho_func)

        assert np.allclose(sds, np.pi/radii, rtol=1e-2)

    def it_can_handle_larger_shapes():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: projector.mathutils.atleast_kd(1/r, r.ndim + 3) * np.random.random_sample(r.shape + (3, 4, 5))
        sds = projector.SurfaceDensity2.calculate(radii, rho_func)

        assert not np.any(np.isnan(sds))

    def it_can_allow_radii_that_are_functions_of_other_parameters():
        def rho_func(r, z):
            z_ = projector.mathutils.atleast_kd(z, r.ndim, append_dims=False)
            return (z_/r**2)[..., None]

        zs = np.linspace(1, 2, 3)
        rs = np.array([
            np.linspace(i, i+1, 10) for i in range(1, zs.size+1)
        ]).T
        sds = projector.SurfaceDensity2.calculate(rs, lambda r: rho_func(r, zs), radial_axis_to_broadcast=1, density_axis=-2)

        zs = projector.mathutils.atleast_kd(zs, rs.ndim, append_dims=False)

        assert np.allclose(sds, (zs * np.pi/rs)[..., None], rtol=1e-2)



def describe_ExcessSurfaceDensity():

    def it_projects_a_constant_density_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: np.ones(r.shape)
        esds = projector.ExcessSurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(esds, np.zeros(radii.shape), atol=1e-3)

    def it_projects_a_simple_power_law_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: 1/r
        esds = projector.ExcessSurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(esds, np.ones(radii.shape), rtol=1e-4)

    def it_can_handle_larger_shapes():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: projector.mathutils.atleast_kd(1/r, r.ndim + 3) * np.random.random_sample(r.shape + (3, 4, 5))
        esds = projector.ExcessSurfaceDensity.calculate(radii, rho_func)

        assert not np.any(np.isnan(esds))

    def it_projects_an_inverse_square_law_correctly():
        radii = np.linspace(0.1, 10, 30)
        rho_func = lambda r: 1/r**2
        esds = projector.ExcessSurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(esds, np.pi/radii, rtol=1e-4)

    def it_projects_a_gaussian_correctly():
        radii = np.logspace(-2, 1, 30)
        rho_func = lambda r: np.exp(-r**2 / 2)
        esds = projector.ExcessSurfaceDensity.calculate(radii, rho_func, num_points=300)

        pref = np.sqrt(2 * np.pi)
        true = pref * (2/radii**2 - np.exp(-radii**2/2) * (1 + 2/radii**2))
        assert np.allclose(esds, true, rtol=1e-2)

    def it_can_allow_radii_that_are_functions_of_other_parameters():
        def rho_func(r, z):
            z_ = projector.mathutils.atleast_kd(z, r.ndim, append_dims=False)
            return (z_/r**2)[..., None]

        zs = np.linspace(1, 2, 3)
        rs = np.array([
            np.linspace(i, i+1, 10) for i in range(1, zs.size+1)
        ]).T
        sds = projector.ExcessSurfaceDensity.calculate(rs, lambda r: rho_func(r, zs), radial_axis_to_broadcast=1)

        zs = projector.mathutils.atleast_kd(zs, rs.ndim, append_dims=False)

        assert np.allclose(sds, (zs * np.pi/rs)[..., None], rtol=1e-4)


def describe_QuadExcessSurfaceDensity():

    def it_projects_a_constant_density_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: np.ones(r.shape)
        esds = projector.QuadExcessSurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(esds, np.zeros(radii.shape), atol=1e-3)

    def it_can_handle_a_constant_in_2d():
        radii = np.linspace(0.1, 10, 2)
        rho_func = lambda r: np.ones(r.shape + (4,))
        esds = projector.QuadExcessSurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(esds, np.zeros(radii.shape + (4,)))

    def it_projects_a_simple_power_law_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: 1/r
        esds = projector.QuadExcessSurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(esds, np.ones(radii.shape), rtol=1e-4)

    def it_projects_an_inverse_square_law_correctly():
        radii = np.linspace(0.1, 10, 30)
        rho_func = lambda r: 1/r**2
        esds = projector.QuadExcessSurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(esds, np.pi/radii, rtol=1e-4)

    def it_can_handle_larger_shapes():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: projector.mathutils.atleast_kd(1/r, r.ndim + 3)
        esds = projector.QuadExcessSurfaceDensity.calculate(radii, rho_func)

        assert np.allclose(esds, np.ones(radii.shape + 3*(1,)), rtol=1e-4)

    def it_complains_when_the_quad_error_gets_big_in_the_first_term():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: np.ones(r.shape) * 1e30
        with pytest.raises(projector.LargeQuadratureErrorsException):
            projector.QuadExcessSurfaceDensity.calculate(radii, rho_func)
