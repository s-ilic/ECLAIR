# FOREGROUNDS V3 (Planck 2018)
import astropy.io.fits as fits
import numpy as np
import itertools

# ------------------------------------------------------------------------------------------------
# Foreground class
# ------------------------------------------------------------------------------------------------
class fgmodel:
    """
    Class of foreground model for the Hillipop likelihood
    Units: Dl in muK^2
    Should return the model in Dl for a foreground emission given the parameters for all correlation of frequencies
    """

    # Planck values

    # MJy.sr-1 -> Kcmb for (100,143,217) GHz
    gnu = {100: 244.059, 143: 371.658, 217: 483.485}

    # frequency dependence (100,143,217,353,545,857) GHz of the SZ effect
    fnu = {100: -4.031, 143: -2.785, 217: 0.187, 353: 6.205, 545: 14.455, 857: 26.335}

    def __init__(self, lmax, freqs, mode="TT", auto=False):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.freqs = freqs
        self.name = None

        # Build the list of cross frequencies
        nfreq = len(freqs)
        self._cross_frequencies = list(
            itertools.combinations_with_replacement(range(nfreq), 2)
            if auto
            else itertools.combinations(range(nfreq), 2)
        )
        pass

    def compute_dl(self, pars):
        """
        Return spectra model for each cross-spectra
        """
        pass


# ------------------------------------------------------------------------------------------------



# Point Sources
class ps(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "PS"
        # Amplitudes of the point sources power spectrum per xfreq
        ell = np.arange(lmax + 1)
        self.ll2pi = ell * (ell + 1) / 2.0 / np.pi

    def compute_dl(self, pars):
        nfreq = len(self.freqs)
        dl_ps = []
        for f1, f2 in self._cross_frequencies:
            freq1 = self.freqs[f1]
            freq2 = self.freqs[f2]
            dl_ps.append( pars["Aps_{}x{}".format(freq1,freq2)] * 1e-6 * self.ll2pi)

        return np.array(dl_ps)



# Radio Point Sources
class ps_radio(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "PS radio"
        # Amplitudes of the point sources power spectrum (MJy^2.sr-1) per xfreq
        # (100x100,100x143,100x217,143x143,143x217,217x217)
        refq = [100, 143, 217]
        radio = [[7.76, 5.36, 4.26], [5.36, 4.83, 3.60], [4.26, 3.60, 3.22]]
        ell = np.arange(lmax + 1)
        ll2pi = ell * (ell + 1) / 2.0 / np.pi
        nfreq = len(freqs)
        self.dl_radio = []
        for f1, f2 in self._cross_frequencies:
            freq1 = freqs[f1]
            freq2 = freqs[f2]
            self.dl_radio.append(
                ll2pi
                * radio[refq.index(freq1)][refq.index(freq2)]
                / self.gnu[freq1]
                / self.gnu[freq2]
            )
        self.dl_radio = np.array(self.dl_radio)

    def compute_dl(self, pars):
        return pars["Aradio"] * self.dl_radio


# Infrared Point Sources
class ps_dusty(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "PS dusty"
        # Amplitudes of the point sources power spectrum (MJy^2.sr-1) per xfreq
        # (100x100,100x143,100x217,143x143,143x217,217x217)
        refq = [100, 143, 217]
        dusty = [
            [0.18179145, 0.47110054, 1.90295795],
            [0.47110054, 1.24375538, 5.06409993],
            [1.90295795, 5.06409993, 20.99528025],
        ]
        ell = np.arange(lmax + 1)
        ll2pi = ell * (ell + 1) / 2.0 / np.pi
        nfreq = len(freqs)
        self.dl_dusty = []
        for f1, f2 in self._cross_frequencies:
            freq1 = freqs[f1]
            freq2 = freqs[f2]
            self.dl_dusty.append(
                ll2pi
                * dusty[refq.index(freq1)][refq.index(freq2)]
                / self.gnu[freq1]
                / self.gnu[freq2]
            )
        self.dl_dusty = np.array(self.dl_dusty)

    def compute_dl(self, pars):
        return pars["Adusty"] * self.dl_dusty


# Dust model
class dust_model(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "Dust"
        self.mode = mode
        self.freqs = freqs

        icol = ["TT","EE","BB","TE","ET"].index(mode)
        
        nfreq = len(freqs)
        self.dl_dust = []
        for f1, f2 in self._cross_frequencies:
            dust_in = fits.getdata("%s_%dx%d.fits" % (filename, freqs[f1], freqs[f2]))
            ell = np.array(dust_in.field(0), int)
            tmpl = np.zeros(max(ell) + 1)
            tmpl[ell] = ell * (ell + 1) / 2.0 / np.pi * dust_in.field(icol+1)
            self.dl_dust.append(tmpl[: lmax + 1])
        self.dl_dust = np.array(self.dl_dust)

    def compute_dl(self, pars):
        if self.mode == "TT":
            A = B = {100:pars["Ad100T"], 143:pars["Ad143T"],217:pars["Ad217T"]}
        if self.mode == "EE":
            A = B = {100:pars["Ad100P"], 143:pars["Ad143P"],217:pars["Ad217P"]}
#            A = B = {100: 0.01703, 143: 0.03605, 217: 0.12498}
        if self.mode == "TE":
            A = {100:pars["Ad100T"], 143:pars["Ad143T"],217:pars["Ad217T"]}
            B = {100:pars["Ad100P"], 143:pars["Ad143P"],217:pars["Ad217P"]}
        if self.mode == "ET":
            A = {100:pars["Ad100P"], 143:pars["Ad143P"],217:pars["Ad217P"]}
            B = {100:pars["Ad100T"], 143:pars["Ad143T"],217:pars["Ad217T"]}
        Ad = []
        for f1, f2 in self._cross_frequencies:
            Ad.append(A[self.freqs[f1]]*B[self.freqs[f2]])
        return np.array(Ad)[:, None] * self.dl_dust


# tSZ (one spectrum for all freqs)
class tsz_model(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        # template: Dl=l(l+1)/2pi Cl, units uK at 143GHz
        self.name = "tSZ"
        nfreq = len(freqs)
        self.dl_sz = []
        dldata = fits.getdata(filename)
        ell = np.array(dldata.ELL, int)
        for f1, f2 in self._cross_frequencies:
            tmpl = np.zeros(max(ell) + 1)
            tmpl[ell] = (
                (dldata.CL1HALO + dldata.CL2HALO)
                * self.fnu[freqs[f1]]
                * self.fnu[freqs[f2]]
                / self.fnu[143]
                / self.fnu[143]
            )
            self.dl_sz.append(tmpl[: lmax + 1])
        self.dl_sz = np.array(self.dl_sz)

    def compute_dl(self, pars):
        return pars["Asz"] * self.dl_sz


# CIB model (one spectrum per xfreq)
class cib_model(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "CIB"
        nfreq = len(freqs)
        self.dl_cib = []
        for f1, f2 in self._cross_frequencies:
            cldata = fits.getdata("%s_%dx%d.fits" % (filename, freqs[f1], freqs[f2]))
            ell = np.array(cldata.ELL, int)
            tmpl = np.zeros(max(ell) + 1)
            ll2pi = ell * (ell + 1) / 2.0 / np.pi
            tmpl[ell] = (
                ll2pi
                * (cldata.CL1HALO + cldata.CL2HALO)
                / self.gnu[freqs[f1]]
                / self.gnu[freqs[f2]]
            )
            self.dl_cib.append(tmpl[: lmax + 1])
        self.dl_cib = np.array(self.dl_cib)

    def compute_dl(self, pars):
        return pars["Acib"] * self.dl_cib


# kSZ (one spectrum for all freqs)
class ksz_model(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        # template: Dl=l(l+1)/2pi Cl, units uK
        self.name = "kSZ"
        nfreq = len(freqs)
        self.dl_ksz = []
        dltemp = fits.getdata(filename)
        ell = np.array(dltemp.ELL, int)
        for f1, f2 in self._cross_frequencies:
            tmp = np.zeros(np.max(ell) + 1)
            tmp[ell] = dltemp.DELL
            self.dl_ksz.append(tmp[: lmax + 1])
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        return pars["Aksz"] * self.dl_ksz


# SZxCIB model (one spectrum per xfreq)
class szxcib_model(fgmodel):
    def __init__(self, lmax, freqs, filename, mode="TT", auto=False):
        super().__init__(lmax, freqs, auto)
        self.name = "SZxCIB"
        nfreq = len(freqs)
        self.dl_szxcib = []
        for f1, f2 in self._cross_frequencies:
            cldata = fits.getdata("%s_%dx%d.fits" % (filename, freqs[f1], freqs[f2]))
            ell = np.array(cldata.ELL, int)
            tmpl = np.zeros(max(ell) + 1)
            ll2pi = ell * (ell + 1) / 2.0 / np.pi
            tmpl[ell] = ll2pi * cldata.CL / self.gnu[freqs[f1]] / self.gnu[freqs[f2]]
            self.dl_szxcib.append(tmpl[: lmax + 1])
        self.dl_szxcib = np.array(self.dl_szxcib)

    def compute_dl(self, pars):
        return pars["Aszxcib"] * self.dl_szxcib

