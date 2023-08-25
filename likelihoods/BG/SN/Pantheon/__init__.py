import os
import sys
import numpy as np
from scipy.linalg import cholesky, solve_triangular

### Path to Pantheon likelihood folder (i.e. the folder this file is in)
path_to_data = os.path.dirname(os.path.realpath(sys.argv[0])) + '/likelihoods/Pantheon/'

### Function loading and reshaping covariance matrices
def load_mat(name):
    tmp = np.loadtxt(name)
    return tmp[1:].reshape((int(tmp[0]), int(tmp[0])))

### Loading the covariance matrices of the measurements
mag_covmat = load_mat(path_to_data + 'sys_full_long.txt')

### Loading the light curve parameters
lc_params = np.genfromtxt(path_to_data + 'lcparam_full_long.txt', names=True)
n_data = len(lc_params)

# Computing the cov mat
cov_mat = mag_covmat + np.diag(lc_params['dmb']**2)

# Computing its Cholesky decomposition
cov_mat = cholesky(cov_mat, lower=True, overwrite_a=True)

### The likelihood function
def get_loglike(class_input, likes_input, class_run):

    # Compute distance moduli
    lum_dist = np.array([class_run.luminosity_distance(z) for z in lc_params['zcmb']])
    dist_mod = 5. * np.log10(lum_dist) + 25.

    # Loading the values of the nuisance parameters for the current chain step
    M = likes_input['M']

    # Computing the difference of data distance moduli - model distance moduli
    diff_dist = lc_params['mb'] - M - dist_mod

    # Solving the triangular system
    diff_dist  = solve_triangular(cov_mat, diff_dist, lower=True, check_finite=False)

    # Return log like
    return -0.5 * (diff_dist**2).sum()
