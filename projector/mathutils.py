import numpy as np


class NoInsertionAxis():
    pass


def atleast_kd(array, k, insertion_axis=NoInsertionAxis()):
    array = np.asarray(array)

    if isinstance(insertion_axis, NoInsertionAxis):
        insertion_axis = array.ndim

    new_shape = array.shape[:insertion_axis] + (1,)*(k-array.ndim) + array.shape[insertion_axis:]

    return array.reshape(new_shape)


def extend_shape(array, k_before, k_after):
    array = np.asarray(array)

    new_shape = k_before*(1,) + array.shape + k_after*(1,)

    return array.reshape(new_shape)


def trapz_(arr, axis, dx=None):
    arr = np.moveaxis(arr, axis, 0)

    if dx is None:
        dx = np.ones(arr.shape[0])
    dx = np.moveaxis(dx, axis, 0)
    dx = atleast_kd(dx, arr.ndim)

    arr = dx*arr

    return 0.5*(arr[0, ...] + 2*arr[1:-1, ...].sum(axis=0) + arr[-1, ...])
