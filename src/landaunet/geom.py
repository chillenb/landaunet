import numpy as np

def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


def copy_halo(data):
    """Copy halo of periodic domain."""
    data[0] = data[-2]
    data[-1] = data[1]
    data[:, 0] = data[:, -2]
    data[:, -1] = data[:, 1]

def neigh_avg(data):
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
            self.data = np.zeros((nx, ny, 3))
        else:
            self.data= data

