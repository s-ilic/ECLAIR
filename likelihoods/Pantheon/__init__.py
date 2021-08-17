import os
import sys
import numexpr
import numpy as np
from scipy.linalg import cholesky, solve_triangular

### Path to Pantheon likelihood folder (i.e. the folder this file is in)
path_to_data = os.path.dirname(os.path.realpath(sys.argv[0])) + '/likelihoods/Pantheon/'

### Function loading and reshaping covariance matrices
def load_mat(name):
    path = path_to_data + name
    tmp = np.loadtxt(path)
    return tmp[1:].reshape((int(tmp[0]), int(tmp[0])))

### Loading the covariance matrices of the measurements
mag_covmat = load_mat('sys_full_long.dat')

### Loading the light curve parameters
lc_params = np.genfromtxt(path_to_data + 'lcparam_full_long.txt', names=True)
n_data = len(lc_params)

### The likelihood function
def get_loglike(class_input, likes_input, class_run):

    # Compute distance moduli
    lum_dist = np.array([class_run.luminosity_distance(z) for z in lc_params['zcmb']])
    dist_mod = 5. * np.log10(lum_dist) + 25.

    # Loading the values of the nuisance parameters for the current chain step
    M = likes_input['M']

    # Computing the difference of data distance moduli - model distance moduli
    diff_dist = lc_params['mb'] - M - dist_mod   

    # Computing the cov mat with nuisance parameters using numexpr for rapidity
    cov_mat = numexpr.evaluate(mag_covmat)
        
    # Add statistical errors to diagonal terms of the cov mat
    cov_mat += mag_covmat + np.diag(lc_params['dmb']**2)
  
    # Whitening the diff_dist :
    # 1)Computing the Cholesky decomposition
    cov_mat = cholesky(cov_mat, lower=True, overwrite_a=True)
    # 2)Solving the triangular system
    diff_dist  = solve_triangular(cov_mat, diff_dist, lower=True, check_finite=False)

    # Return log like
    return -0.5 * (diff_dist**2).sum()

