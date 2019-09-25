import pytest
import numpy as np
import projector


def describe_esd():

    @pytest.fixture
    def esd_args():
        return np.linspace(0.1, 10, 10), lambda r: 1/r**3

    def it_is_fast_for_50_points(esd_args, benchmark):
        benchmark(projector.esd, *esd_args, num_points=50)

    def it_is_fast_for_100_points(esd_args, benchmark):
        benchmark(projector.esd, *esd_args, num_points=100)

    def it_is_fast_for_200_points(esd_args, benchmark):
        benchmark(projector.esd, *esd_args, num_points=200)
