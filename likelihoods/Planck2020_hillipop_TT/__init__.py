import glob
import logging
import os
import re
from itertools import combinations
from typing import Optional

import astropy.io.fits as fits
import numpy as np

from . import foregrounds as fg
from . import tools


#############################
### First initialisations ###
#############################

planck_2020_root = os.environ.get('PLANCK_2020_DATA')
if planck_2020_root == None:
    raise ValueError('The environment variable PLANCK_2020_DATA is not set.')

likelihood_name = "TT"
data_folder = planck_2020_root + "/hillipop"
multipoles_range_file = "data/binning_v4.2.fits"
xspectra_basename = "data/dl_PR4_v4.2"
covariance_matrix_file = "data/invfll_PR4_v4.2_TT.fits"

fg_list = {
    "sbpx": fg.subpix,
    "ps": fg.ps,
    "dust": fg.dust,
    "dust_model": fg.dust_model,
    "sync": fg.sync_model,
    "ksz": fg.ksz_model,
    "ps_radio": fg.ps_radio,
    "ps_dusty": fg.ps_dusty,
    "cib": fg.cib_model,
    "tsz": fg.tsz_model,
    "szxcib": fg.szxcib_model,
}

foregrounds = {
    'TT': {
        'dust_model': 'foregrounds/DUST_Planck_PR4_model_v4.2',
        'tsz': 'foregrounds/SZ_Planck_PR4_model.txt',
        'ksz': 'foregrounds/kSZ_Planck_PR4_model.txt',
        'cib': 'foregrounds/CIB_Planck_PR4_model.txt',
        'szxcib': 'foregrounds/SZxCIB_Planck_PR4_model.txt',
        'ps_radio': None,
        'ps_dusty': None,
    },
    'EE': {
        'dust_model': 'foregrounds/DUST_Planck_PR4_model_v4.2',
    },
    'TE': {
        'dust_model': 'foregrounds/DUST_Planck_PR4_model_v4.2',
    },
}

frequencies = [100, 100, 143, 143, 217, 217]
_mapnames = ["100A", "100B", "143A", "143B", "217A", "217B"]
_nmap = len(frequencies)
_nfreq = len(np.unique(frequencies))
_nxfreq = _nfreq * (_nfreq + 1) // 2
_nxspec = _nmap * (_nmap - 1) // 2

# Get likelihood name and add the associated mode
likelihood_modes = [likelihood_name[i:i+2] for i in range(0,len(likelihood_name),2)]
_is_mode = {mode: mode in likelihood_modes for mode in ["TT", "TE", "EE"]}
_is_mode["ET"] = _is_mode["TE"]


##########################
### Auxilary functions ###
##########################

def _xspec2xfreq():
    list_fqs = []
    for f1 in range(_nfreq):
        for f2 in range(f1, _nfreq):
            list_fqs.append((f1, f2))

    freqs = list(np.unique(frequencies))
    spec2freq = []
    for m1 in range(_nmap):
        for m2 in range(m1 + 1, _nmap):
            f1 = freqs.index(frequencies[m1])
            f2 = freqs.index(frequencies[m2])
            spec2freq.append(list_fqs.index((f1, f2)))

    return spec2freq

def _set_multipole_ranges(filename):
    """
    Return the (lmin,lmax) for each cross-spectra for each mode (TT, EE, TE, ET)
    array(nmode,nxspec)
    """
    if not os.path.exists(filename):
        raise ValueError("File missing {}".format(filename))

    tags = ["TT", "EE", "BB", "TE"]
    lmins = {}
    lmaxs = {}
    for tag in ["TT","EE","TE"]:
        hdu = tags.index(tag)
        data = fits.getdata(filename, hdu + 1) # file HDU [TT,EE,BB,TE]
        lmins[tag] = np.array(data.field(0), int)
        lmaxs[tag] = np.array(data.field(1), int)
    lmins["ET"] = lmins["TE"]
    lmaxs["ET"] = lmaxs["TE"]

    return lmins, lmaxs

def _read_dl_xspectra(basename, hdu=1):
    """
    Read xspectra from Xpol [Dl in K^2]
    Output: Dl (TT,EE,TE,ET) in muK^2
    """

    nhdu = len( fits.open(f"{basename}_{_mapnames[0]}x{_mapnames[1]}.fits"))
    if nhdu < hdu:
        #no sig in file, uniform weight
        dldata = np.ones( (_nxspec, 4, lmax+1))
    else:
        if nhdu == 1: hdu=0 #compatibility
        dldata = []
        for m1, m2 in combinations(_mapnames, 2):
            data = fits.getdata( f"{basename}_{m1}x{m2}.fits", hdu)*1e12
            tmpcl = list(data[[0,1,3],:lmax+1])
            data = fits.getdata( f"{basename}_{m2}x{m1}.fits", hdu)*1e12
            tmpcl.append( data[3,:lmax+1])
            dldata.append( tmpcl)

    dldata = np.transpose(np.array(dldata), (1, 0, 2))
    return dict(zip(['TT','EE','TE','ET'],dldata))


def _read_invcovmatrix(filename):
    """
    Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
    Output: invkll [Dl in muK^-4]
    """
    if not os.path.exists(filename):
        raise ValueError("File missing {}".format(filename))

    data = fits.getdata(filename)
    nel = int(np.sqrt(data.size))
    data = data.reshape((nel, nel)) / 1e24  # muK^-4

    nell = _get_matrix_size()
    if nel != nell:
        raise ValueError("Incoherent covariance matrix (read:%d, expected:%d)" % (nel, nell))

    return data

def _get_matrix_size():
    """
    Compute covariance matrix size given activated mode
    Return: number of multipole
    """
    nell = 0

    # TT,EE,TEET
    for m in ["TT", "EE", "TE"]:
        if _is_mode[m]:
            nells = _lmaxs[m] - _lmins[m] + 1
            nell += np.sum([nells[_xspec2xfreq.index(k)] for k in range(_nxfreq)])

    return nell


##############################
### Second initialisations ###
##############################

_xspec2xfreq = _xspec2xfreq()

# Multipole ranges
filename = os.path.join(data_folder, multipoles_range_file)
_lmins, _lmaxs = _set_multipole_ranges(filename)
lmax = np.max([max(l) for l in _lmaxs.values()])

# Data
basename = os.path.join(data_folder, xspectra_basename)
_dldata = _read_dl_xspectra(basename)

# Weights
dlsig = _read_dl_xspectra(basename, hdu=2)
for m,w8 in dlsig.items(): w8[w8==0] = np.inf
_dlweight = {k:1/v**2 for k,v in dlsig.items()}

# Inverted Covariance matrix
filename = os.path.join(data_folder, covariance_matrix_file)
# Sanity check
m = re.search(".*_(.+?).fits", covariance_matrix_file)
if not m or likelihood_name != m.group(1):
    raise ValueError(
        "The covariance matrix mode differs from the likelihood mode. Check the given path [%s]" % covariance_matrix_file
    )
_invkll = _read_invcovmatrix(filename)
_invkll = _invkll.astype('float32')

# Foregrounds
fgs = {}  # list of foregrounds per mode [TT,EE,TE,ET]
# Init foregrounds TT
fgsTT = []
if _is_mode["TT"]:
    for name in foregrounds["TT"].keys():
        if name not in fg_list.keys():
            raise ValueError("Unknown foreground model '%s'!" % name)

        kwargs = dict(lmax=lmax, freqs=frequencies, mode="TT")
        if isinstance(foregrounds["TT"][name], str):
            kwargs["filename"] = os.path.join(data_folder, foregrounds["TT"][name])
        elif name == "szxcib":
            filename_tsz = foregrounds["TT"]["tsz"] and os.path.join(data_folder, foregrounds["TT"]["tsz"])
            filename_cib = foregrounds["TT"]["cib"] and os.path.join(data_folder, foregrounds["TT"]["cib"])
            kwargs["filenames"] = (filename_tsz,filename_cib)
        fgsTT.append(fg_list[name](**kwargs))
fgs['TT'] = fgsTT

# Init foregrounds EE
fgsEE = []
if _is_mode["EE"]:
    for name in foregrounds["EE"].keys():
        if name not in fg_list.keys():
            raise ValueError("Unknown foreground model '%s'!" % name)
        kwargs = dict(lmax=lmax, freqs=frequencies)
        if isinstance(foregrounds["EE"][name], str):
            kwargs["filename"] = os.path.join(data_folder, foregrounds["EE"][name])
        fgsEE.append(fg_list[name](mode="EE", **kwargs))
fgs['EE'] = fgsEE

# Init foregrounds TE
fgsTE = []
fgsET = []
if _is_mode["TE"]:
    for name in foregrounds["TE"].keys():
        if name not in fg_list.keys():
            raise ValueError("Unknown foreground model '%s'!" % name)
        kwargs = dict(lmax=lmax, freqs=frequencies)
        if isinstance(foregrounds["TE"][name], str):
            kwargs["filename"] = os.path.join(data_folder, foregrounds["TE"][name])
        fgsTE.append(fg_list[name](mode="TE", **kwargs))
        fgsET.append(fg_list[name](mode="ET", **kwargs))
fgs['TE'] = fgsTE
fgs['ET'] = fgsET


##################################
### Auxilary functions (cont.) ###
##################################

def _select_spectra(cl, mode):
    """
    Cut spectra given Multipole Ranges and flatten
    Return: list
    """
    acl = np.asarray(cl)
    xl = []
    for xf in range(_nxfreq):
        lmin = _lmins[mode][_xspec2xfreq.index(xf)]
        lmax = _lmaxs[mode][_xspec2xfreq.index(xf)]
        xl += list(acl[xf, lmin : lmax + 1])
    return xl

def _xspectra_to_xfreq(cl, weight, normed=True):
    """
    Average cross-spectra per cross-frequency
    """
    xcl = np.zeros((_nxfreq, lmax + 1))
    xw8 = np.zeros((_nxfreq, lmax + 1))
    for xs in range(_nxspec):
        xcl[_xspec2xfreq[xs]] += weight[xs] * cl[xs]
        xw8[_xspec2xfreq[xs]] += weight[xs]

    xw8[xw8 == 0] = np.inf
    if normed:
        return xcl / xw8
    else:
        return xcl, xw8

def _compute_residuals(pars, dlth, mode):

    # Nuisances
    cal = []
    for m1, m2 in combinations(_mapnames, 2):
            if mode == "TT":
                cal1, cal2 = pars[f"cal{m1}"], pars[f"cal{m2}"]
            elif mode == "EE":
                cal1, cal2 = pars[f"cal{m1}"]*pars[f"pe{m1}"], pars[f"cal{m2}"]*pars[f"pe{m2}"]
            elif mode == "TE":
                cal1, cal2 = pars[f"cal{m1}"], pars[f"cal{m2}"]*pars[f"pe{m2}"]
            elif mode == "ET":
                cal1, cal2 = pars[f"cal{m1}"]*pars[f"pe{m1}"], pars[f"cal{m2}"]
            cal.append(cal1 * cal2 / pars["A_planck"] ** 2)

    # Data
    dldata = _dldata[mode]

    # Model
    dlmodel = [dlth[mode]] * _nxspec
    for fg in fgs[mode]:
        dlmodel += fg.compute_dl(pars)

    # Compute Rl = Dl - Dlth
    Rspec = np.array([dldata[xs] - cal[xs] * dlmodel[xs] for xs in range(_nxspec)])

    return Rspec

def dof():
    return len(_invkll)

# def reduction_matrix(mode):
#     """
#     Reduction matrix

#     each column is equal to 1 in the 15 elements corresponding to a cross-power spectrum
#     measurement in that multipole and zero elsewhere

#     """
#     X = np.zeros( (len(delta_cl),lmax+1) )
#     x0 = 0
#     for xf in range(_nxfreq):
#         lmin = _lmins[mode][_xspec2xfreq.index(xf)]
#         lmax = _lmaxs[mode][_xspec2xfreq.index(xf)]
#         for il,l in enumerate(range(lmin,lmax+1)):
#             X[x0+il,l] = 1
#         x0 += (lmax-lmin+1)
    
#     return X

def compute_chi2(dlth, params_values):
    """
    Compute likelihood from model out of Boltzmann code
    Units: Dl in muK^2

    Parameters
    ----------
    pars: dict
            parameter values
    dl: array or arr2d
            CMB power spectrum (Dl in muK^2)

    Returns
    -------
    lnL: float
        Log likelihood for the given parameters -2ln(L)
    """

    # cl_boltz from Boltzmann (Cl in muK^2)
    # lth = np.arange(lmax + 1)
    # dlth = np.asarray(dl)[:, lth][[0, 1, 3, 3]]  # select TT,EE,TE,TE

    # Create Data Vector
    Xl = []
    if _is_mode["TT"]:
        # compute residuals Rl = Dl - Dlth
        Rspec = _compute_residuals(params_values, dlth, "TT")
        # average to cross-spectra
        Rl = _xspectra_to_xfreq(Rspec, _dlweight["TT"])
        # select multipole range
        Xl += _select_spectra(Rl, "TT")

    if _is_mode["EE"]:
        # compute residuals Rl = Dl - Dlth
        Rspec = _compute_residuals(params_values, dlth, "EE")
        # average to cross-spectra
        Rl = _xspectra_to_xfreq(Rspec, _dlweight["EE"])
        # select multipole range
        Xl += _select_spectra(Rl, "EE")

    if _is_mode["TE"] or _is_mode["ET"]:
        Rl = 0
        Wl = 0
        # compute residuals Rl = Dl - Dlth
        if _is_mode["TE"]:
            Rspec = _compute_residuals(params_values, dlth, "TE")
            RlTE, WlTE = _xspectra_to_xfreq(Rspec, _dlweight["TE"], normed=False)
            Rl = Rl + RlTE
            Wl = Wl + WlTE
        if _is_mode["ET"]:
            Rspec = _compute_residuals(params_values, dlth, "ET")
            RlET, WlET = _xspectra_to_xfreq(Rspec, _dlweight["ET"], normed=False)
            Rl = Rl + RlET
            Wl = Wl + WlET
        # select multipole range
        Xl += _select_spectra(Rl / Wl, 'TE')

    delta_cl = np.asarray(Xl).astype('float32')
    # chi2 = self.delta_cl @ self._invkll @ self.delta_cl
    chi2 = _invkll.dot(delta_cl).dot(delta_cl)

    return chi2

def get_loglike(class_input, likes_input, class_run):
    dl = class_run.lensed_cl()
    for s in ['tt', 'ee', 'te']:
        dl[s] *= (1e12 * class_run.T_cmb()**2. *
                  dl['ell'] * (dl['ell'] + 1) / 2. / np.pi)
    lth = np.arange(lmax + 1)
    dlth = {k.upper():dl[k][:lmax+1] for k in dl.keys()}
    dlth['ET'] = dlth['TE']

    chi2 = compute_chi2(dlth, likes_input)

    return -0.5 * chi2
