# Bin class
import numpy as np


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
        self.lmin = np.min(self.lmins)
        self.lmax = np.max(self.lmaxs)
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

    def bin_spectra(self, spectra):
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by `l(l+1)/2pi`.
        Return Cb
        """
        spectra = np.asarray(spectra)
        minlmax = np.min([spectra.shape[-1] - 1, self.lmax])

        _p, _q = self._bin_operators()
        return np.dot(spectra[..., : minlmax + 1], _p.T[: minlmax + 1, ...])

    def bin_covariance(self, clcov):
        p, q = self._bin_operators(cov=True)
        return np.matmul(p, np.matmul(clcov, q))
