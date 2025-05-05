import os
import numpy as np
import astropy.io.fits as fits


############
# SETTINGS #
############
default_ell_lims = {
    "hillipop_TTTEEE_ell_min_TT": 30,
    "hillipop_TTTEEE_ell_max_TT": 2500,
    "hillipop_TTTEEE_ell_min_EE": 30,
    "hillipop_TTTEEE_ell_max_EE": 2000,
    "hillipop_TTTEEE_ell_min_TE": 30,
    "hillipop_TTTEEE_ell_max_TE": 2000,
}
##########

planck_pr4_root = os.environ.get('PLANCK_PR4_DATA')
likelihood_name = "TTTEEE"
data_folder = planck_pr4_root + "/hillipop"
multipoles_range_file = "data/binning_v4.2.fits"
xspectra_basename = "data/dl_PR4_v4.2"
covariance_matrix_file = "data/invfll_PR4_v4.2_TTTEEE.fits"

frequencies = [100, 100, 143, 143, 217, 217]
_mapnames = ["100A", "100B", "143A", "143B", "217A", "217B"]
_nmap = len(frequencies)
_nfreq = len(np.unique(frequencies))
_nxfreq = _nfreq * (_nfreq + 1) // 2
_nxspec = _nmap * (_nmap - 1) // 2

likelihood_modes = [likelihood_name[i:i+2] for i in range(0,len(likelihood_name),2)]
_is_mode = {mode: mode in likelihood_modes for mode in ["TT", "TE", "EE"]}
_is_mode["ET"] = _is_mode["TE"]

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

_xspec2xfreq = _xspec2xfreq()

filename = os.path.join(data_folder, multipoles_range_file)
_lmins, _lmaxs = _set_multipole_ranges(filename)
lmax = np.max([max(l) for l in _lmaxs.values()])

def _get_ell_vector(mode):
    all_ell = []
    for xf in range(_nxfreq):
        lmin = _lmins[mode][_xspec2xfreq.index(xf)]
        lmax = _lmaxs[mode][_xspec2xfreq.index(xf)]
        all_ell += list(range(lmin, lmax + 1))
    return all_ell

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

    return data

def create_cut_covmat(lkl_input):

    # Define ell limits
    ell_lims = {}
    for typ in ["min", "max"]:
        for mode in ["TT", "EE", "TE"]:
            if f'hillipop_TTTEEE_ell_{typ}_{mode}' in lkl_input:
                ell_lims[f'ell_{typ}_{mode}'] = lkl_input[f'hillipop_TTTEEE_ell_{typ}_{mode}']
            else:
                ell_lims[f'ell_{typ}_{mode}'] = default_ell_lims[f'hillipop_TTTEEE_ell_{typ}_{mode}']

    # Check ell limits
    for mode in ["TT", "EE", "TE"]:
        if ell_lims[f'ell_min_{mode}'] < 30:
            raise ValueError(f"hillipop_TTTEEE_ell_min_{mode} must be >= 30")
        maxi = 2500 if mode == "TT" else 2000
        if ell_lims[f'ell_max_{mode}'] > maxi:
            raise ValueError(f"hillipop_TTTEEE_ell_max_{mode} must be <= {maxi}")
        if ell_lims[f'ell_min_{mode}'] > ell_lims[f'ell_max_{mode}']:
            raise ValueError(f"hillipop_TTTEEE_ell_min_{mode} must be <= hillipop_TTTEEE_ell_max_{mode}")

    full_cov_fname = os.path.join(data_folder, "data/cov_PR4_v4.2_TTTEEE.npy")
    if os.path.isfile(full_cov_fname):
        cov = np.load(full_cov_fname)
    else:
        filename = os.path.join(data_folder, covariance_matrix_file)
        icov = _read_invcovmatrix(filename)
        #_invkll = _invkll.astype('float32')
        cov = np.linalg.inv(icov)
        np.save(full_cov_fname,  cov)

    all_ell_range = []
    if _is_mode["TT"]:
        all_ell_range += [(ell, 'TT') for ell in _get_ell_vector("TT")]
    if _is_mode["EE"]:
        all_ell_range += [(ell, 'EE') for ell in _get_ell_vector("EE")]
    if _is_mode["TE"] or _is_mode["ET"]:
        all_ell_range += [(ell, 'TE') for ell in _get_ell_vector("TE")]
    keep_ix_ell_range = []
    for ix, (l, m) in enumerate(all_ell_range):
        if ell_lims[f'ell_min_{m}'] <= l <= ell_lims[f'ell_max_{m}']:
            keep_ix_ell_range.append(ix)

    partial_icov_fname = "data/icov_PR4_v4.2_TTTEEE"
    partial_icov_fname += f"_TT-{ell_lims['ell_min_TT']}-{ell_lims['ell_max_TT']}"
    partial_icov_fname += f"_EE-{ell_lims['ell_min_EE']}-{ell_lims['ell_max_EE']}"
    partial_icov_fname += f"_TE-{ell_lims['ell_min_TE']}-{ell_lims['ell_max_TE']}"
    partial_icov_fname += ".npy"
    partial_icov_fname = os.path.join(data_folder, partial_icov_fname)

    partial_icov = np.linalg.inv(cov[:, keep_ix_ell_range][keep_ix_ell_range, :])
    np.save(partial_icov_fname, partial_icov)
