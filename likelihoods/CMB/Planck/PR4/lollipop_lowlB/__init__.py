import os
from typing import Optional

import astropy.io.fits as fits
import numpy as np

from . import tools
from .bins import Bins

#######################
### Initialisations ###
#######################

planck_pr4_root = os.environ.get('PLANCK_PR4_DATA')
if planck_pr4_root == None:
    raise ValueError('The environment variable PLANCK_PR4_DATA is not set.')

mode = "lowlB"
cl_file = 'cl_lolEB_NPIPE.dat'
fiducial_file = 'fiducial_lolEB_planck2018_tensor_lensedCls.dat'
cl_cov_file = 'clcov_lolEB_NPIPE.fits'
hartlap_factor = False
marginalised_over_covariance = True
Nsim = 400
lmin = 2
lmax = 30

bins = tools.get_binning(lmin, lmax)

data = tools.read_dl(planck_pr4_root + '/lollipop/' + cl_file)
cldata = bins.bin_spectra(data)

data = tools.read_dl(planck_pr4_root + '/lollipop/' + fiducial_file)
clfid = bins.bin_spectra(data)

clcov = fits.getdata(planck_pr4_root + '/lollipop/' + cl_cov_file)
if mode == "lowlEB":
    cbcov = tools.bin_covEB(clcov, bins)
elif mode == "lowlE":
    cbcov = tools.bin_covEE(clcov, bins)
elif mode == "lowlB":
    cbcov = tools.bin_covBB(clcov, bins)
clvar = np.diag(cbcov).reshape(-1, bins.nbins)

if mode == "lowlEB":
    rcond = 1e-9 #getattr(self, "rcond", 1e-9)
    invclcov = np.linalg.pinv(cbcov, rcond)
else:
    invclcov = np.linalg.inv(cbcov)

if hartlap_factor:
    if Nsim != 0:
        invclcov *= (Nsim - len(cbcov) - 2) / (Nsim - 1)

if marginalised_over_covariance:
    if Nsim <= 1:
        raise ValueError(
            "Need the number of MC simulations used to compute the covariance in order to marginalise over (Nsim>1).",
        )

fsky = 0.52 #getattr(self, "fsky", 0.52)
cloff = tools.compute_offsets(bins.lbin, clvar, clfid, fsky=fsky)
cloff[2:] = 0.0  # force NO offsets EB

##########################
### Auxilary functions ###
##########################

def _compute_chi2_2fields(cl, params_values):
    """
    Compute offset-Hamimeche&Lewis likelihood
    Input: Cl in muK^2
    """
    # get model in Cl, muK^2
    clth = np.array(
        [bins.bin_spectra(cl[mode]) for mode in ["ee", "bb", "eb"] if mode in cl]
    )

    cal = params_values["A_planck"]**2

    nell = cldata.shape[1]
    x = np.zeros(cldata.shape)
    for ell in range(nell):
        O = tools.vec2mat(cloff[:, ell])
        D = tools.vec2mat(cldata[:, ell] * cal) + O
        M = tools.vec2mat(clth[:, ell]) + O
        F = tools.vec2mat(clfid[:, ell]) + O

        # compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
        w, V = np.linalg.eigh(M)
        #            if prod( sign(w)) <= 0:
        #                print( "WARNING: negative eigenvalue for l=%d" %l)
        L = V @ np.diag(1.0 / np.sqrt(w)) @ V.transpose()
        P = L.transpose() @ D @ L

        # apply HL transformation
        w, V = np.linalg.eigh(P)
        g = np.sign(w) * tools.ghl(np.abs(w))
        G = V @ np.diag(g) @ V.transpose()

        # cholesky fiducial
        w, V = np.linalg.eigh(F)
        L = V @ np.diag(np.sqrt(w)) @ V.transpose()

        # compute C_fid^1/2 * G * C_fid^1/2
        X = L.transpose() @ G @ L
        x[:, ell] = tools.mat2vec(X)

    # compute chi2
    x = x.flatten()
    if marginalised_over_covariance:
        chi2 = Nsim * np.log(1 + (x @ invclcov @ x) / (Nsim - 1))
    else:
        chi2 = x @ invclcov @ x

    return chi2

def _compute_chi2_1field(cl, params_values):
    """
    Compute offset-Hamimeche&Lewis likelihood
    Input: Cl in muK^2
    """
    # model in Cl, muK^2
    m = 0 if mode == "lowlE" else 1
    clth = bins.bin_spectra(cl["ee" if mode == "lowlE" else "bb"])

    cal = params_values["A_planck"]**2

    x = (cldata[m] * cal + cloff[m]) / (clth + cloff[m])
    g = np.sign(x) * tools.ghl(np.abs(x))

    X = (np.sqrt(clfid[m] + cloff[m])) * g * (np.sqrt(clfid[m] + cloff[m]))

    if marginalised_over_covariance:
        # marginalised over S = Ceff
        chi2 = Nsim * np.log(1 + (X @ invclcov @ X) / (Nsim - 1))
    else:
        chi2 = X @ invclcov @ X

    return chi2

################################
### Main likelihood function ###
################################

def get_loglike(class_input, likes_input, class_run):
    cl = class_run.lensed_cl()
    for s in ['tt', 'ee', 'te', 'bb']:
        cl[s] *= 1e12 * class_run.T_cmb()**2.
    if 'eb' not in cl.keys():
        cl['eb'] = cl['ee'].copy() * 0.
    if mode == "lowlEB":
        chi2 = _compute_chi2_2fields(cl, likes_input)
    elif mode in ["lowlE", "lowlB"]:
        chi2 = _compute_chi2_1field(cl, likes_input)
    return -0.5 * chi2
