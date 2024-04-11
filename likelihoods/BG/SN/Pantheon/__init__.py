import os
import sys
import numpy as np
from scipy.linalg import cholesky, solve_triangular

class likelihood:
  def __init__(self, lkl_input):
    ### Path to Pantheon likelihood folder (i.e. the folder this file is in)
    path_to_data = os.path.dirname(os.path.realpath(sys.argv[0]))
    path_to_data += '/likelihoods/BG/SN/Pantheon/'

    ### Loading the covariance matrices of the measurements
    tmp = np.loadtxt(path_to_data + 'sys_full_long.txt')
    mag_covmat = tmp[1:].reshape((int(tmp[0]), int(tmp[0])))

    ### Loading the light curve parameters
    self.lc_params = np.genfromtxt(path_to_data + 'lcparam_full_long.txt',
                                   names=True)
    n_data = len(self.lc_params)

    # Computing the cov mat
    self.cov_mat = mag_covmat + np.diag(self.lc_params['dmb']**2)

    # Computing its Cholesky decomposition
    self.cov_mat = cholesky(self.cov_mat, lower=True, overwrite_a=True)

  def get_loglike(self, class_input, lkl_input, class_run):
    # Compute distance moduli
    lum_dist = [class_run.luminosity_distance(z) for z in self.lc_params['zcmb']]
    lum_dist = np.array(lum_dist)
    dist_mod = 5. * np.log10(lum_dist) + 25.

    # Loading the values of the nuisance parameters for the current chain step
    M = lkl_input['M']

    # Computing the difference of data distance moduli - model distance moduli
    diff_dist = self.lc_params['mb'] - M - dist_mod

    # Solving the triangular system
    diff_dist  = solve_triangular(self.cov_mat, diff_dist,
                                  lower=True, check_finite=False)

    # Return log like
    return -0.5 * (diff_dist**2).sum()
