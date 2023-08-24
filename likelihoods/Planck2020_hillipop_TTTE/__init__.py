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

likelihood_name = "TTTE"
data_folder = planck_2020_root + "/hillipop"
multipoles_range_file = "data/binning_ext.fits"
xspectra_basename = "data/cross_NPIPE_detset"
covariance_matrix_file = "data/invfll_NPIPE_detset_ext_TTTE.fits"

fg_list = {
    "ps": fg.ps,
    "dust": fg.dust_model,
    "ksz": fg.ksz_model,
    "ps_radio": fg.ps_radio,
    "ps_dusty": fg.ps_dusty,
    "cib": fg.cib_model,
    "tsz": fg.tsz_model,
    "szxcib": fg.szxcib_model,
}

foregrounds = {
    "TT": {
        "dust": "foregrounds/DUST_planck_npipe_model",
        "tsz": "foregrounds/SZ_planck_npipe_model.fits",
        "ksz": "foregrounds/kSZ_planck_npipe_model.fits",
        "cib": "foregrounds/CIB_planck_npipe_model_v3",
        "szxcib": "foregrounds/SZxCIB_planck_npipe_model",
        "ps_radio": None,
        "ps_dusty": None,
        "ps": None,
    },
    "EE": {
        "dust": "foregrounds/DUST_planck_npipe_model",
    },
    "TE": {
        "dust": "foregrounds/DUST_planck_npipe_model",
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

    lmins = []
    lmaxs = []
    for hdu in [0, 1, 3, 3]:  # file HDU [TT,EE,BB,TE]
        data = fits.getdata(filename, hdu + 1)
        lmins.append(np.array(data.field(0), int))
        lmaxs.append(np.array(data.field(1), int))

    return lmins, lmaxs

def _read_dl_xspectra(basename, field=1):
    """
    Read xspectra from Xpol [Dl in K^2]
    Output: Dl in muK^2
    """

    dldata = []
    for m1, m2 in combinations(range(_nmap), 2):
        tmpcl = []
        for mode, hdu in {"TT": 1, "EE": 2, "TE": 4, "ET": 4}.items():
            filename = "{}_{}_{}.fits".format(basename, m1, m2)
            if mode == "ET":
                filename = "{}_{}_{}.fits".format(basename, m2, m1)
            if not os.path.exists(filename):
                raise ValueError("File missing {}".format(filename))
            data = fits.getdata(filename, hdu)
            ell = np.array(data.field(0), int)
            datacl = np.zeros(np.max(ell) + 1)
            datacl[ell] = data.field(field) * 1e12
            tmpcl.append(datacl[: lmax + 1])

        dldata.append(tmpcl)

    return np.transpose(np.array(dldata), (1, 0, 2))

def _read_invcovmatrix(filename):
    """
    Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
    Output: invkll [Dl in muK^-4]
    """
    if not os.path.exists(filename):
        raise ValueError("File missing {}".format(filename))

#        data = fits.getdata(filename).field(0)
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
    for im, m in enumerate(["TT", "EE", "TE"]):
        if _is_mode[m]:
            nells = _lmaxs[im] - _lmins[im] + 1
            nell += np.sum([nells[_xspec2xfreq.index(k)] for k in range(_nxfreq)])

    return nell


##############################
### Second initialisations ###
##############################

_xspec2xfreq = _xspec2xfreq()

# Multipole ranges
filename = os.path.join(data_folder, multipoles_range_file)
_lmins, _lmaxs = _set_multipole_ranges(filename)
lmax = np.max([max(l) for l in _lmaxs])

# Data
basename = os.path.join(data_folder, xspectra_basename)
_dldata = _read_dl_xspectra(basename, field=1)

# Weights
dlsig = _read_dl_xspectra(basename, field=2)
dlsig[dlsig == 0] = np.inf
_dlweight = 1.0 / dlsig ** 2

# Inverted Covariance matrix
filename = os.path.join(data_folder, covariance_matrix_file)
# Sanity check
m = re.search(".*_(.+?).fits", covariance_matrix_file)
if not m or likelihood_name != m.group(1):
    raise ValueError(
        "The covariance matrix mode differs from the likelihood mode. Check the given path [%s]" % covariance_matrix_file
    )
_invkll = _read_invcovmatrix(filename)

# Foregrounds
fgs = []  # list of foregrounds per mode [TT,EE,TE,ET]
# Init foregrounds TT
fgsTT = []
if _is_mode["TT"]:
    for name in foregrounds["TT"].keys():
        if name not in fg_list.keys():
            raise ValueError("Unknown foreground model '%s'!" % name)

        kwargs = dict(lmax=lmax, freqs=frequencies, mode="TT")
        if isinstance(foregrounds["TT"][name], str):
            kwargs["filename"] = os.path.join(data_folder, foregrounds["TT"][name])
        fgsTT.append(fg_list[name](**kwargs))
fgs.append(fgsTT)

# Init foregrounds EE
fgsEE = []
if _is_mode["EE"]:
    for name in foregrounds["EE"].keys():
        if name not in fg_list.keys():
            raise ValueError("Unknown foreground model '%s'!" % name)
        filename = os.path.join(data_folder, foregrounds["EE"].get(name))
        fgsEE.append(
            fg_list[name](lmax, frequencies, mode="EE", filename=filename)
        )
fgs.append(fgsEE)

# Init foregrounds TE
fgsTE = []
fgsET = []
if _is_mode["TE"]:
    for name in foregrounds["TE"].keys():
        if name not in fg_list.keys():
            raise ValueError("Unknown foreground model '%s'!" % name)
        filename = os.path.join(data_folder, foregrounds["TE"].get(name))
        kwargs = dict(lmax=lmax, freqs=frequencies, filename=filename)
        fgsTE.append(fg_list[name](mode="TE", **kwargs))
        fgsET.append(fg_list[name](mode="ET", **kwargs))
fgs.append(fgsTE)
fgs.append(fgsET)


##################################
### Auxilary functions (cont.) ###
##################################

def _select_spectra(cl, mode=0):
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

def _compute_residuals(pars, dlth, mode=0):

    # Nuisances
    cal = []
    for m1, m2 in combinations(range(_nmap), 2):
        cal.append(pars["A_planck"] ** 2 * (1. + pars["cal%s" % _mapnames[m1]] + pars["cal%s" % _mapnames[m2]]))

    # Data
    dldata = _dldata[mode]

    # Model
    dlmodel = [dlth[mode]] * _nxspec
    for fg in fgs[mode]:
        dlmodel += fg.compute_dl(pars)

    # Compute Rl = Dl - Dlth
    Rspec = np.array([dldata[xs] - cal[xs] * dlmodel[xs] for xs in range(_nxspec)])

    return Rspec

def compute_chi2(dl, params_values):
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
    lth = np.arange(lmax + 1)
    dlth = np.asarray(dl)[:, lth][[0, 1, 3, 3]]  # select TT,EE,TE,TE

    # Create Data Vector
    Xl = []
    if _is_mode["TT"]:
        # compute residuals Rl = Dl - Dlth
        Rspec = _compute_residuals(params_values, dlth, mode=0)
        # average to cross-spectra
        Rl = _xspectra_to_xfreq(Rspec, _dlweight[0])
        # select multipole range
        Xl += _select_spectra(Rl, mode=0)

    if _is_mode["EE"]:
        # compute residuals Rl = Dl - Dlth
        Rspec = _compute_residuals(params_values, dlth, mode=1)
        # average to cross-spectra
        Rl = _xspectra_to_xfreq(Rspec, _dlweight[1])
        # select multipole range
        Xl += _select_spectra(Rl, mode=1)

    if _is_mode["TE"] or _is_mode["ET"]:
        Rl = 0
        Wl = 0
        # compute residuals Rl = Dl - Dlth
        if _is_mode["TE"]:
            Rspec = _compute_residuals(params_values, dlth, mode=2)
            RlTE, WlTE = _xspectra_to_xfreq(Rspec, _dlweight[2], normed=False)
            Rl = Rl + RlTE
            Wl = Wl + WlTE
        if _is_mode["ET"]:
            Rspec = _compute_residuals(params_values, dlth, mode=3)
            RlET, WlET = _xspectra_to_xfreq(Rspec, _dlweight[3], normed=False)
            Rl = Rl + RlET
            Wl = Wl + WlET
        # select multipole range
        Xl += _select_spectra(Rl / Wl, mode=2)

    Xl = np.array(Xl)
    chi2 = Xl @ _invkll @ Xl

    return chi2

def get_loglike(class_input, likes_input, class_run):
    dl = class_run.lensed_cl()
    for s in ['tt', 'ee', 'te']:
        dl[s] *= (1e12 * class_run.T_cmb()**2. *
                  dl['ell'] * (dl['ell'] + 1) / 2. / np.pi)
    lth = np.arange(lmax + 1)
    dlth = np.zeros((4, lmax + 1))
    dlth[0] = dl["tt"][lth]
    dlth[1] = dl["ee"][lth]
    dlth[3] = dl["te"][lth]

    chi2 = compute_chi2(dlth, likes_input)

    return -0.5 * chi2
