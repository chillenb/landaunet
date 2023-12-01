import numpy as np

def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


def copy_halo(data):
    """Copy halo of periodic domain."""
    data[0] = data[-2]
    data[-1] = data[1]
    data[:, 0] = data[:, -2]
    data[:, -1] = data[:, 1]

class Periodic2D:
    """Periodic 2D domain."""

    def __init__(self, nx, ny, h, D, data=None):
        self.nx = nx
        self.ny = ny
        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError("nx and ny must be even.")
        self.h = h
        self.D = D

        if np.asarray(D).size == 1:
            self.anisotropic = False
        else:
            self.anisotropic = True

        if data is None:
            self.data = np.zeros((nx+2, ny+2, 3))
        else:
            self.data = np.zeros((nx+2, ny+2, 3))
            self.data[1:-1, 1:-1] = data
            copy_halo(self.data)

