import numpy as np
from landaunet import _core

def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


def copy_halo(data):
    """Copy halo of periodic domain."""
    data[0] = data[-2]
    data[-1] = data[1]
    data[:, 0] = data[:, -2]
    data[:, -1] = data[:, 1]

def neigh_avg(data, dirichletbc=False):
    return (np.roll(data, 1, axis=0) + np.roll(data, -1, axis=0) + \
              np.roll(data, 1, axis=1) + np.roll(data, -1, axis=1)) * 0.25

def skewvecs(vecs, coef=1.0):
    skv = np.zeros((vecs.shape[0], vecs.shape[1], 3, 3))
    
    np.multiply(vecs[..., 2], coef, out=skv[..., 0, 1])
    np.multiply(vecs[..., 1], -coef, out=skv[..., 0, 2])
    np.multiply(vecs[..., 2], -coef, out=skv[..., 1, 0])
    np.multiply(vecs[..., 0], coef, out=skv[..., 1, 2])
    np.multiply(vecs[..., 1], coef, out=skv[..., 2, 0])
    np.multiply(vecs[..., 0], -coef, out=skv[..., 2, 1])
    

    return skv

def red_black_isotropic_step(P, Q, dt):
    skewPavg = skewvecs(neigh_avg(P), coef=dt/2)
    Q = Q + (skewPavg @ Q[...,None])[...,0]
    Q = np.linalg.solve(np.eye(3)-skewPavg, Q)
    
    skewQnewavg = skewvecs(neigh_avg(Q), coef=dt/2)
    P = P + (skewQnewavg  @ P[...,None])[...,0]
    P = np.linalg.solve(np.eye(3)-skewQnewavg, P)
    return P, Q

def red_black_isotropic_start(data, dt):
    ck = checkerboard(data.shape[:2])
    P = data.copy()
    Q = data.copy()
    P[ck==1] = 0
    Q[ck==0] = 0
    skewPavg = skewvecs(neigh_avg(P), coef=dt/2)
    Qhalf = np.linalg.solve(np.eye(3)-skewPavg, Q)
    Qhalf[ck==0] = 0
    return P, Qhalf

def reassemble(P, Q):
    ck = checkerboard(P.shape[:2])
    data = np.zeros_like(P)
    data[ck==0] = P[ck==0]
    data[ck==1] = Q[ck==1]
    return data

def red_black_isotropic_end(P, Q, dt):
    skewPavg = skewvecs(neigh_avg(P), coef=dt/2)
    Qnext = ((np.eye(3)+skewPavg) @ Q[...,None])[...,0]
    return reassemble(P, Qnext)


def halo_copy(data, dirichletbc=False):
    """Copy halo of periodic domain."""
    if dirichletbc is not None:
        data[:, 0] = data[:, -2]
        data[:, -1] = data[:, 1]
        data[:, :, 0] = data[:, :, -2]
        data[:, :, -1] = data[:, :, 1]
    else:
        data[:, 0, :] = dirichletbc[:, None]
        data[:, -1, :] = dirichletbc[:, None]
        data[:, :, 0] = dirichletbc[:, None]
        data[:, :, -1] = dirichletbc[:, None]


class Periodic2D:
    """Periodic 2D domain."""

    def __init__(self, nx, ny, dt, D, data=None):
        self.nx = nx
        self.ny = ny
        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError("nx and ny must be even.")
        self.dt = dt
        self.D = D

        if np.asarray(D).size == 1:
            self.anisotropic = False
        else:
            self.anisotropic = True

        self.data = np.zeros((3, nx + 2, ny + 2))
        if data is not None:
            self.data[:, 1:-1, 1:-1] = data
            halo_copy(self.data, dirichletbc=np.array([0.0,0.0,1.0]))

    def setup_data(self):
        P, Q = red_black_isotropic_start(np.moveaxis(self.data, 0, 2), self.dt)
        self.data = reassemble(P, Q)
        self.data = np.moveaxis(self.data, 2, 0).copy()
        halo_copy(self.data)
    
    def step(self):
        _core.redblack_step_2d(self.data, self.dt)

def normalizedata(data):
    return data / np.linalg.norm(data, axis=0)[None, ...]

class UnitNormVec3RF:
    def __init__(self,  N=12, mostly_z = None):

        self.N = N

        self.freqs = 2*np.pi*np.arange(-N*1.0, N+1.0, 1.0)
        self.fsize = 2*N+1
        self.mostly_z = mostly_z

    def random(self, size):
        magnitudes = (np.random.randn(size, 3 * (self.fsize)**2) +
                      1j*np.random.randn(size, 3 * (self.fsize)**2)) \
                      * 1/(2*self.fsize)**2

        return magnitudes

    def eval_batch(self, features, xs):
        x_tr = np.exp(1j*np.outer(xs[:,0], self.freqs))
        y_tr = np.exp(1j*np.outer(xs[:,1], self.freqs))

        f = np.reshape(features, (-1, self.freqs.size, self.freqs.size, 3))
        prod = np.einsum('abcd, qb, qc -> aqd', f, x_tr, y_tr).real
        prod = np.array(prod/np.linalg.norm(prod, axis=-1)[..., None])
        if self.mostly_z is not None:
            prod = prod * (1-self.mostly_z)
            prod[:,:,-1] += 1
            prod = np.array(prod/np.linalg.norm(prod, axis=-1)[..., None])

        return prod