import numpy as np
from numpy import linalg

from .bins import Bins


def compute_offsets(ell, varcl, clref, fsky=1.0, iter=10):
    Nl = np.sqrt(np.abs(varcl - (2.0 / (2.0 * ell + 1) * clref ** 2) / fsky))
    for i in range(iter):
        Nl = np.sqrt(np.abs(varcl - 2.0 / (2.0 * ell + 1) / fsky * (clref ** 2 + 2.0 * Nl * clref)))
    return Nl * np.sqrt((2.0 * ell + 1) / 2.0)


def read_dl(datafile):
    data = np.loadtxt(datafile).T
    dl = np.zeros((3, int(max(data[0])) + 1))  # EE,BB,EB
    l = np.array(data[0], int)
    dl[0, l] = data[1]
    dl[1, l] = data[2]
    dl[2, l] = data[3]
    return dl


def get_binning():
    dl = 10
    llmin = 2
    llmax = 35
    hlmin = 36
    hlmax = 150
    lmins = list(range(llmin, llmax + 1)) + list(range(hlmin, hlmax - dl + 2, dl))
    lmaxs = list(range(llmin, llmax + 1)) + list(range(hlmin + dl - 1, hlmax + 1, dl))
    binc = Bins(lmins, lmaxs)
    return binc


def bin_covEB(clcov, binc):
    nell = len(clcov) // 3
    cbcov = np.zeros((3 * binc.nbins, 3 * binc.nbins))
    for t1 in range(3):
        for t2 in range(3):
            mymat = np.zeros((binc.lmax + 1, binc.lmax + 1))
            mymat[2:, 2:] = clcov[
                t1 * nell : t1 * nell + (binc.lmax - 1), t2 * nell : t2 * nell + (binc.lmax - 1)
            ]
            cbcov[
                t1 * binc.nbins : (t1 + 1) * binc.nbins, t2 * binc.nbins : (t2 + 1) * binc.nbins
            ] = binc.bin_covariance(mymat)
    return cbcov


def bin_covBB(clcov, binc):
    nell = len(clcov) // 3
    t1 = t2 = 1
    mymat = np.zeros((binc.lmax + 1, binc.lmax + 1))
    mymat[2:, 2:] = clcov[
        t1 * nell : t1 * nell + (binc.lmax - 1), t2 * nell : t2 * nell + (binc.lmax - 1)
    ]
    cbcov = binc.bin_covariance(mymat)
    return cbcov


def bin_covEE(clcov, binc):
    nell = len(clcov) // 3
    t1 = t2 = 0
    mymat = np.zeros((binc.lmax + 1, binc.lmax + 1))
    mymat[2:, 2:] = clcov[
        t1 * nell : t1 * nell + (binc.lmax - 1), t2 * nell : t2 * nell + (binc.lmax - 1)
    ]
    cbcov = binc.bin_covariance(mymat)
    return cbcov


def vec2mat(vect):
    """
    shape EE, BB and EB as a matrix
    input:
        vect: EE,BB,EB
    output:
        matrix: [[EE,EB],[EB,BB]]
    """
    mat = np.zeros((2, 2))
    mat[0, 0] = vect[0]
    mat[1, 1] = vect[1]
    if len(vect) == 3:
        mat[1, 0] = mat[0, 1] = vect[2]
    return mat


def mat2vec(mat):
    """
    shape polar matrix into polar vect
    input:
        matrix: [[EE,EB],[EB,BB]]
    output:
        vect: EE,BB,EB
    """
    vec = np.array([mat[0, 0], mat[1, 1], mat[0, 1]])
    return vec


def ghl(x):
    return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))
