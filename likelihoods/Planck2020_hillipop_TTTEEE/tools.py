# ------------------------------------------------------------------------------------------------
# Hillipop External Tools
# ------------------------------------------------------------------------------------------------
import os

import astropy.io.fits as fits
import numpy as np
import scipy.ndimage as nd
from numpy.linalg import *

tagnames = ["TT", "EE", "TE", "ET"]



# ------------------------------------------------------------------------------------------------
# def create_bin_file(filename, lbinTT, lbinEE, lbinBB, lbinTE, lbinET)
# def SG(l, cl, nsm=5, lcut=0)
# def convert_to_stdev(sigma)
# def ctr_level(histo2d, lvl)
# class Bins(object)
# ------------------------------------------------------------------------------------------------


def create_bin_file(filename, lbinTT, lbinEE, lbinBB, lbinTE, lbinET):
    """
    lbin = [(lmin,lmax)] for each 15 cross-spectra
    """
    h = fits.Header()
    hdu = [fits.PrimaryHDU(header=h)]

    def fits_layer(lbin):
        h = fits.Header()
        lmin = np.array([l[0] for l in lbin])
        lmax = np.array([l[1] for l in lbin])
        c1 = fits.Column(name="LMIN", array=lmin, format="1D")
        c2 = fits.Column(name="LMAX", array=lmax, format="1D")
        return fits.BinTableHDU.from_columns([c1, c2], header=h)

    hdu.append(fits_layer(lbinTT))
    hdu.append(fits_layer(lbinEE))
    hdu.append(fits_layer(lbinBB))
    hdu.append(fits_layer(lbinTE))
    hdu.append(fits_layer(lbinET))

    hdulist = fits.HDUList(hdu)
    hdulist.writeto(filename, overwrite=True)


# smooth cls before Cov computation
def SG(l, cl, nsm=5, lcut=0):
    clSG = np.copy(cl)

    # gauss filter
    if lcut < 2 * nsm:
        shift = 0
    else:
        shift = 2 * nsm

    data = nd.gaussian_filter1d(clSG[max(0, lcut - shift) :], nsm)
    clSG[lcut:] = data[shift:]

    return clSG


def convert_to_stdev(sigma):
    """
    Given a grid of likelihood values, convert them to cumulative
    standard deviation.  This is useful for drawing contours from a
    grid of likelihoods.
    """
    #    sigma = np.exp(-logL+np.max(logL))

    shape = sigma.shape
    sigma = sigma.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(sigma)[::-1]
    i_unsort = np.argsort(i_sort)

    sigma_cumsum = sigma[i_sort].cumsum()
    sigma_cumsum /= sigma_cumsum[-1]

    return sigma_cumsum[i_unsort].reshape(shape)


def ctr_level(histo2d, lvl):
    """
    Extract the contours for the 2d plots
    """

    h = histo2d.flatten() * 1.0
    h.sort()
    cum_h = np.cumsum(h[::-1])
    cum_h /= cum_h[-1]

    alvl = np.searchsorted(cum_h, lvl)
    clist = h[-alvl]

    return clist


# ------------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------------
# Binning
# ------------------------------------------------------------------------------------------------
class Bins(object):
    """
    lmins : list of integers
        Lower bound of the bins
    lmaxs : list of integers
        Upper bound of the bins
    """

    def __init__(self, lmins, lmaxs):
        if not (len(lmins) == len(lmaxs)):
            raise ValueError("Incoherent inputs")

        lmins = np.asarray(lmins)
        lmaxs = np.asarray(lmaxs)
        cutfirst = np.logical_and(lmaxs >= 2, lmins >= 2)
        self.lmins = lmins[cutfirst]
        self.lmaxs = lmaxs[cutfirst]

        self._derive_ext()

    @classmethod
    def fromdeltal(cls, lmin, lmax, delta_ell):
        nbins = (lmax - lmin + 1) // delta_ell
        lmins = lmin + np.arange(nbins) * delta_ell
        lmaxs = lmins + delta_ell - 1
        return cls(lmins, lmaxs)

    def _derive_ext(self):
        for l1, l2 in zip(self.lmins, self.lmaxs):
            if l1 > l2:
                raise ValueError("Incoherent inputs")
        self.lmin = min(self.lmins)
        self.lmax = max(self.lmaxs)
        if self.lmin < 1:
            raise ValueError("Input lmin is less than 1.")
        if self.lmax < self.lmin:
            raise ValueError("Input lmax is less than lmin.")

        self.nbins = len(self.lmins)
        self.lbin = (self.lmins + self.lmaxs) / 2.0
        self.dl = self.lmaxs - self.lmins + 1

    def bins(self):
        return (self.lmins, self.lmaxs)

    def cut_binning(self, lmin, lmax):
        sel = np.where((self.lmins >= lmin) & (self.lmaxs <= lmax))[0]
        self.lmins = self.lmins[sel]
        self.lmaxs = self.lmaxs[sel]
        self._derive_ext()

    def _bin_operators(self, Dl=False, cov=False):
        if Dl:
            ell2 = np.arange(self.lmax + 1)
            ell2 = ell2 * (ell2 + 1) / (2 * np.pi)
        else:
            ell2 = np.ones(self.lmax + 1)
        p = np.zeros((self.nbins, self.lmax + 1))
        q = np.zeros((self.lmax + 1, self.nbins))

        for b, (a, z) in enumerate(zip(self.lmins, self.lmaxs)):
            dl = z - a + 1
            p[b, a : z + 1] = ell2[a : z + 1] / dl
            if cov:
                q[a : z + 1, b] = 1 / ell2[a : z + 1] / dl
            else:
                q[a : z + 1, b] = 1 / ell2[a : z + 1]

        return p, q

    def bin_spectra(self, spectra, Dl=False):
        """
        Average spectra with defined bins
        can be weighted by `l(l+1)/2pi`.
        Return Cb
        """
        spectra = np.asarray(spectra)
        minlmax = min([spectra.shape[-1] - 1, self.lmax])

        _p, _q = self._bin_operators(Dl=Dl)
        return np.dot(spectra[..., : minlmax + 1], _p.T[: minlmax + 1, ...])

    def bin_covariance(self, clcov):
        """
        Average covariance with defined bins
        """
        p, q = self._bin_operators(cov=True)
        return np.matmul(p, np.matmul(clcov, q))
