import pytest
import numpy as np
import projector


def describe_esd():

    def it_projects_a_constant_density_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: np.ones(r.shape)
        esds = projector.esd(radii, rho_func)

        assert np.allclose(esds, np.zeros(radii.shape), atol=1e-3)

    def it_projects_a_simple_power_law_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: 1/r
        esds = projector.esd(radii, rho_func)

        assert np.allclose(esds, np.ones(radii.shape), rtol=1e-4)

    def it_can_handle_larger_shapes():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: projector.mathutils.atleast_kd(1/r, r.ndim + 3) * np.random.random_sample(r.shape + (3, 4, 5))
        esds = projector.esd(radii, rho_func)

        assert not np.any(np.isnan(esds))


def describe_esd_quad():

    def it_projects_a_constant_density_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: np.ones(r.shape)
        esds = projector.esd_quad(radii, rho_func)

        assert np.allclose(esds, np.zeros(radii.shape), atol=1e-3)

    def it_can_handle_a_constant_in_2d():
        radii = np.linspace(0.1, 10, 2)
        rho_func = lambda r: np.ones(r.shape + (4,))
        esds = projector.esd_quad(radii, rho_func)

        assert np.allclose(esds, np.zeros(radii.shape + (4,)))

    def it_projects_a_simple_power_law_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: 1/r
        esds = projector.esd_quad(radii, rho_func)

        assert np.allclose(esds, np.ones(radii.shape), rtol=1e-4)

    def it_can_handle_larger_shapes():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: projector.mathutils.atleast_kd(1/r, r.ndim + 3)
        esds = projector.esd_quad(radii, rho_func)

        assert np.allclose(esds, np.ones(radii.shape + 3*(1,)), rtol=1e-4)

    def it_complains_when_the_quad_error_gets_big_in_the_first_term():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: np.ones(r.shape) * 1e30
        with pytest.raises(projector.LargeQuadratureErrorsException):
            projector.esd_quad(radii, rho_func)
