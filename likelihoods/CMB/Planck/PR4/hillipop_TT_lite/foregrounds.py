# FOREGROUNDS V4.2 (Planck PR4)
import astropy.io.fits as fits
import os
import numpy as np
import itertools

t_cmb = 2.72548
k_b = 1.3806503e-23
h_pl = 6.626068e-34

# ------------------------------------------------------------------------------------------------
# Foreground class
# ------------------------------------------------------------------------------------------------
class fgmodel:
    """
    Class of foreground model for the Hillipop likelihood
    Units: Dl in muK^2
    Should return the model in Dl for a foreground emission given the parameters for all correlation of frequencies
    """

    #reference frequency for residuals amplitudes
    f0 = 143

    # Planck effective frequencies
    fsz    = {100:100.24, 143: 143, 217: 222.044}
    fdust  = {100:105.2, 143:147.5, 217:228.1, 353:370.5} #alpha=4 from [Planck 2013 IX]
    fcib   = fdust
    fsyn   = {100:100,143:143,217:217}
    fradio = {100:100.4,143:140.5,217:218.6}

    def _f_tsz( self, freq):
        # Freq in GHz
        nu = freq*1e9
        xx=h_pl*nu/(k_b*t_cmb)
        return xx*( 1/np.tanh(xx/2.) ) - 4

    def _f_Planck( self, f, T):
        # Freq in GHz
        nu = f*1e9
        xx  = h_pl*nu /(k_b*T)
        return (nu**3.)/(np.exp(xx)-1.)

    #Temp Antenna conversion
    def _dBdT(self, f):
        # Freq in GHz
        nu  = f*1e9
        xx  = h_pl*nu /(k_b*t_cmb)
        return (nu)**4 * np.exp(xx) / (np.exp(xx)-1.)**2.

    def _tszRatio( self, f, f0):
        return self._f_tsz(f)/self._f_tsz(f0)

    def _cibRatio( self, f, f0, beta=1.75, T=25):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )

    def _dustRatio( self, f, f0, beta=1.5, T=19.6):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )

    def _radioRatio( self, f, f0, beta=-0.7):
        return (f/f0)**beta / ( self._dBdT(f)/self._dBdT(f0) )

    def _syncRatio( self, f, f0, beta=-0.7):
        return (f/f0)**beta / ( self._dBdT(f)/self._dBdT(f0) )

    def __init__(self, lmax, freqs, mode="TT", auto=False, **kwargs):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.freqs = freqs
        self.name = None

        ell = np.arange(lmax + 1)
        self.ll2pi = ell * (ell + 1) / (3000*3001)

        # Build the list of cross frequencies
        self._cross_frequencies = list(
            itertools.combinations_with_replacement(freqs, 2)
            if auto
            else itertools.combinations(freqs, 2)
        )
        pass

    def _gen_dl_powerlaw( self, alpha, lnorm=3000):
        """
        Generate power-law Dl template
        Input: alpha in Cl
        """
        lmax = self.lmax if lnorm is None else max(self.lmax,lnorm)
        ell = np.arange( 2, lmax+1)

        template = np.zeros( lmax+1)
        template[np.array(ell,int)] = ell*(ell+1)/2/np.pi * ell**(alpha)

        #normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]

        return template[:self.lmax+1]

    def _read_dl_template( self, filename, lnorm=3000):
        """
        Read FG template (in Dl, muK^2)
        WARNING: need to check file before reading...
        """

        if not os.path.exists(filename):
            raise ValueError("Missing file: %s" % self.filename)

        #read dl template
        l,data = np.loadtxt( filename, unpack=True)
        l = np.array(l,int)

        if max(l) < self.lmax:
            print( "WARNING: template {} has lower lmax (filled with 0)".format(filename))
        template = np.zeros( max(self.lmax,max(l)) + 1)
        template[l] = data

        #normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]
        
        return template[:self.lmax+1]

    def compute_dl(self, pars):
        """
        Return spectra model for each cross-spectra
        """
        pass
# ------------------------------------------------------------------------------------------------



# Subpixel effect
class subpix(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "SubPixel"
        self.fwhm = {100:9.68,143:7.30,217:5.02} #arcmin

    def compute_dl(self, pars):
        def _bl( fwhm):
            sigma = np.deg2rad(fwhm/60.) / np.sqrt(8.0 * np.log(2.0))
            ell = np.arange(self.lmax + 1)
            return np.exp(-0.5 * ell * (ell + 1) * sigma**2)

        dl_sbpx = []
        for f1, f2 in self._cross_frequencies:
            pxl = self.ll2pi / _bl( self.fwhm[f1]) / _bl( self.fwhm[f2])
            dl_sbpx.append( pars["Asbpx_{}x{}".format(f1,f2)] * pxl / pxl[2500] )

        if self.mode == "TT":
            return np.array(dl_sbpx)
        else:
            return 0.



# Point Sources
class ps(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS"

    def compute_dl(self, pars):
        dl_ps = []
        for f1, f2 in self._cross_frequencies:
            dl_ps.append( pars["Aps_{}x{}".format(f1,f2)] * self.ll2pi)

        if self.mode == "TT":
            return np.array(dl_ps)
        else:
            return 0.



# Radio Point Sources (v**alpha)
class ps_radio(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS radio"

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi
                * self._radioRatio( self.fradio[f1], self.f0, beta=pars['beta_radio'])
                * self._radioRatio( self.fradio[f2], self.f0, beta=pars['beta_radio'])
            )

        if self.mode == "TT":
            return pars["Aradio"] * np.array(dl)
        else:
            return 0.


# Infrared Point Sources
class ps_dusty(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS dusty"

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi
                * self._cibRatio( self.fcib[f1], self.f0, beta=pars['beta_dusty'])
                * self._cibRatio( self.fcib[f2], self.f0, beta=pars['beta_dusty'])
            )

        if self.mode == "TT":
            return pars["Adusty"] * np.array(dl)
        else:
            return 0.


# Galactic Dust
class dust(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Dust"
        
        self.dlg = []
        hdr = ["ell","100x100","100x143","100x217","143x143","143x217","217x217"]
        data = np.loadtxt( f"{filename}_{mode}.txt").T
        l = np.array(data[0],int)
        for f1, f2 in self._cross_frequencies:
            tmpl = np.zeros(max(l) + 1)
            tmpl[l] = data[hdr.index(f"{f1}x{f2}")]
            self.dlg.append( tmpl[:lmax+1])

        self.dlg = np.array(self.dlg)

    def compute_dl(self, pars):
        if self.mode == "TT":
            A = B = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}
        if self.mode == "EE":
            A = B = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
        if self.mode == "TE":
            A = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}
            B = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
        if self.mode == "ET":
            A = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
            B = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}

        Ad = [A[f1]*B[f2] for f1, f2 in self._cross_frequencies]

        return np.array(Ad)[:, None] * self.dlg


class dust_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Dust model"
        
        self.dlg = []
        hdr = ["ell","100x100","100x143","100x217","143x143","143x217","217x217"]
        data = np.loadtxt( f"{filename}_{mode}.txt").T
        l = np.array(data[0],int)
        for f1, f2 in self._cross_frequencies:
            tmpl = np.zeros(max(l) + 1)
            tmpl[l] = data[hdr.index(f"{f1}x{f2}")]
            self.dlg.append( tmpl[:lmax+1])
        self.dlg = np.array(self.dlg)

    def compute_dl(self, pars):
        if   self.mode == "TT": beta1,beta2 = pars['beta_dustT'],pars['beta_dustT']
        elif self.mode == "TE": beta1,beta2 = pars['beta_dustT'],pars['beta_dustP']
        elif self.mode == "ET": beta1,beta2 = pars['beta_dustP'],pars['beta_dustT']
        elif self.mode == "EE": beta1,beta2 = pars['beta_dustP'],pars['beta_dustP']

        if   self.mode == "TT": ad1,ad2 = pars['AdustT'],pars['AdustT']
        elif self.mode == "TE": ad1,ad2 = pars['AdustT'],pars['AdustP']
        elif self.mode == "ET": ad1,ad2 = pars['AdustP'],pars['AdustT']
        elif self.mode == "EE": ad1,ad2 = pars['AdustP'],pars['AdustP']

        dl = []
        for xf, (f1, f2) in enumerate(self._cross_frequencies):
            dl.append( ad1 * ad2 * self.dlg[xf]
                       * self._dustRatio( self.fdust[f1], self.fdust[353], beta=beta1)
                       * self._dustRatio( self.fdust[f2], self.fdust[353], beta=beta2)
                       )
        return np.array(dl)


# Syncrothron model
class sync_model(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Synchrotron"

        #check effective freqs
        for f in freqs:
            if f not in self.fsyn:
                raise ValueError( f"Missing SYNC effective frequency for {f}")

        alpha_syn = -2.5  #Cl template power-law TBC
        self.dl_syn = self._gen_dl_powerlaw( alpha_syn, lnorm=100)
        self.beta_syn = -0.7

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_syn
                       * self._syncRatio( self.fsyn[f1], self.f0, beta=self.beta_syn)
                       * self._syncRatio( self.fsyn[f2], self.f0, beta=self.beta_syn)
                       )
        if self.mode == "TT":
            return pars["AsyncT"] * np.array(dl)
        elif self.mode == "EE":
            return pars["AsyncP"] * np.array(dl)
        else:
            return 0.


# CIB model
class cib_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "clustered CIB"

        #check effective freqs
        for f in freqs:
            if f not in self.fcib:
                raise ValueError( f"Missing CIB effective frequency for {f}")

        if filename is None:
            alpha_cib = -1.3
            self.dl_cib = self._gen_dl_powerlaw( alpha_cib)
        else:
            self.dl_cib = self._read_dl_template( filename)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_cib
                       * self._cibRatio( self.fcib[f1], self.f0, beta=pars['beta_cib'])
                       * self._cibRatio( self.fcib[f2], self.f0, beta=pars['beta_cib'])
                       )
        if self.mode == "TT":
            return pars["Acib"] * np.array(dl)
        else:
            return 0.

# tSZ (one spectrum for all freqs)
class tsz_model(fgmodel):
    def __init__(self, lmax, freqs, filename="", mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        # template: Dl=l(l+1)/2pi Cl, units uK at 143GHz
        self.name = "tSZ"

        #check effective freqs for SZ
        for f in freqs:
            if f not in self.fsz:
                raise ValueError( f"Missing SZ effective frequency for {f}")

        # read Dl template (normalized at l=3000)
        sztmpl = self._read_dl_template(filename)

        self.dl_sz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_sz.append( sztmpl[: lmax + 1]
                               * self._tszRatio( self.fsz[f1], self.f0)
                               * self._tszRatio( self.fsz[f2], self.f0)
                               )
        self.dl_sz = np.array(self.dl_sz)

    def compute_dl(self, pars):
        return pars["Atsz"] * self.dl_sz


# kSZ
class ksz_model(fgmodel):
    def __init__(self, lmax, freqs, filename="", mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        # template: Dl=l(l+1)/2pi Cl, units uK
        self.name = "kSZ"

        # read Dl template (normalized at l=3000)
        ksztmpl = self._read_dl_template(filename)

        self.dl_ksz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(ksztmpl[: lmax + 1])
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            return pars["Aksz"] * self.dl_ksz
        else:
            return 0.


# SZxCIB model
class szxcib_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False, **kwargs):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "SZxCIB"

        #check effective freqs for SZ
        for f in freqs:
            if f not in self.fsz:
                raise ValueError( f"Missing SZ effective frequency for {f}")

        #check effective freqs for dust
        for f in freqs:
            if f not in self.fcib:
                raise ValueError( f"Missing Dust effective frequency for {f}")

        self._is_template = filename
        if self._is_template:
            self.x_tmpl = self._read_dl_template(filename)
        elif "filenames" in kwargs:
            self.x_tmpl = self._read_dl_template(kwargs["filenames"][0])*self._read_dl_template(kwargs["filenames"][1])
        else:
            raise ValueError( f"Missing template for SZxCIB")
            
    def compute_dl(self, pars):
        dl_szxcib = []
        for f1, f2 in self._cross_frequencies:
            dl_szxcib.append( self.x_tmpl * (
                self._tszRatio(self.fsz[f2],self.f0) * self._cibRatio(self.fcib[f1],self.f0,beta=pars['beta_cib']) +
                self._tszRatio(self.fsz[f1],self.f0) * self._cibRatio(self.fcib[f2],self.f0,beta=pars['beta_cib'])
                )
            )

        if self.mode == "TT":
            return -1. * pars["xi"] * np.sqrt(pars["Acib"]*pars["Atsz"]) * np.array(dl_szxcib)
        else:
            return 0.
