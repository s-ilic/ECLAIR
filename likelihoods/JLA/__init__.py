import os
import sys
import numexpr
import numpy as np
from scipy.linalg import cholesky, solve_triangular

### Path to JLA likelihood folder (i.e. the folder this file is in)
path_to_data = os.path.dirname(os.path.realpath(sys.argv[0])) + '/JAM_likelihoods/JLA/'

### Function loading and reshaping covariance matrices
def load_mat(name):
    path = path_to_data + name
    tmp = np.loadtxt(path)
    return tmp[1:].reshape((int(tmp[0]), int(tmp[0])))

### Loading the covariance matrices of the measurements
mag_covmat = load_mat('jla_v0_covmatrix.dat')
stretch_covmat = load_mat('jla_va_covmatrix.dat')
colour_covmat = load_mat('jla_vb_covmatrix.dat')
mag_stretch_covmat = load_mat('jla_v0a_covmatrix.dat')
mag_colour_covmat = load_mat('jla_v0b_covmatrix.dat')
stretch_colour_covmat = load_mat('jla_vab_covmatrix.dat')

### Loading the light curve parameters
lc_params = np.genfromtxt(path_to_data + 'jla_lcparams.txt', names=True)
n_data = len(lc_params)

### Setting the scriptmcut parameter from jla.dataset file
scriptmcut = 10.0

### The likelihood function
def get_loglike(class_input, likes_input, class_run):

    # Compute distance moduli
    lum_dist = np.array([class_run.luminosity_distance(z) for z in lc_params['zcmb']])
    dist_mod = 5. * np.log10(lum_dist) + 25.

    # Loading the values of the nuisance parameters for the current chain step
    alpha = likes_input['alpha']
    beta = likes_input['beta']
    M = likes_input['M']
    Delta_M = likes_input['Delta_M']

    # Computing the difference of data distance moduli - model distance moduli
    diff_dist = (
        lc_params['mb'] - (M - alpha * lc_params['x1'] + beta * lc_params['color']
        + Delta_M * (lc_params['3rdvar'] > scriptmcut)) - dist_mod
    )

    # Computing the cov mat with nuisance parameters using numexpr for rapidity
    cov_mat = numexpr.evaluate(
        "(mag_covmat + alpha**2 * stretch_covmat + beta**2 * colour_covmat"
        "+ 2. * alpha * mag_stretch_covmat - 2. * beta * mag_colour_covmat"
        "- 2. * alpha * beta * stretch_colour_covmat)"
    )

    # Add statistical errors to diagonal terms of the cov mat
    cov_mat += (
        np.diag(lc_params['dmb']**2 + (alpha * lc_params['dx1'])**2
        + (beta * lc_params['dcolor'])**2 + 2. * alpha * lc_params['cov_m_s']
        - 2. * beta * lc_params['cov_m_c'] - 2. * alpha * beta * lc_params['cov_s_c'])
    )

    # Whitening the diff_dist :
    # 1)Computing the Cholesky decomposition
    cov_mat = cholesky(cov_mat, lower=True, overwrite_a=True)
    # 2)Solving the triangular system
    diff_dist  = solve_triangular(cov_mat, diff_dist, lower=True, check_finite=False)

    # Return log like
    return -0.5 * (diff_dist**2).sum()
