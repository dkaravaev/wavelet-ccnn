import numpy as np


def tochanneled(s):
    r = np.real(s)
    i = np.imag(s)
    cs = np.hstack((r.reshape(s.shape[0], 1), i.reshape(s.shape[0], 1)))
    return np.float32(cs.reshape(cs.shape[0], cs.shape[1]))

def tochanneled_m(s):
    return np.hstack((np.real(s), np.imag(s)))

def tocomplex(s):
    return s[:, 0] + 1j * s[:, 1]
