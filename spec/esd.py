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
