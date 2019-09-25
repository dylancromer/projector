import numpy as np
import projector


def describe_esd():

    def it_projects_a_constant_density_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: 1
        esds = projector.esd(radii, rho_func)

        assert np.allclose(esds, np.zeros(radii.shape), atol=1e-3)

    def it_projects_a_simple_power_law_correctly():
        radii = np.linspace(0.1, 10, 10)
        rho_func = lambda r: 1/r
        esds = projector.esd(radii, rho_func)

        assert np.allclose(esds, np.ones(radii.shape), rtol=1e-3)
