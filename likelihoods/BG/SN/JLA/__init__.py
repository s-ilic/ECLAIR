import os
import sys
import numexpr
import numpy as np
from scipy.linalg import cholesky, solve_triangular

class likelihood:
  def __init__(self, lkl_input):
    # Path to JLA likelihood folder (i.e. the folder this file is in)
    path_to_data = os.path.dirname(os.path.realpath(sys.argv[0]))
    path_to_data += '/likelihoods/BG/SN/JLA/'

    # Function loading and reshaping covariance matrices
    def load_mat(name):
      path = path_to_data + name
      tmp = np.loadtxt(path)
      return tmp[1:].reshape((int(tmp[0]), int(tmp[0])))

    # Loading the covariance matrices of the measurements
    self.mag_covmat = load_mat('jla_v0_covmatrix.dat')
    self.stretch_covmat = load_mat('jla_va_covmatrix.dat')
    self.colour_covmat = load_mat('jla_vb_covmatrix.dat')
    self.mag_stretch_covmat = load_mat('jla_v0a_covmatrix.dat')
    self.mag_colour_covmat = load_mat('jla_v0b_covmatrix.dat')
    self.stretch_colour_covmat = load_mat('jla_vab_covmatrix.dat')

    # Loading the light curve parameters
    self.lc_params = np.genfromtxt(path_to_data + 'jla_lcparams.txt',
                                   names=True)

    # Setting the scriptmcut parameter from jla.dataset file
    self.scriptmcut = 10.0

  def get_loglike(self, class_input, lkl_input, class_run):
    # Compute distance moduli
    lum_dist = [class_run.luminosity_distance(z) for z in self.lc_params['zcmb']]
    lum_dist = np.array(lum_dist)
    dist_mod = 5. * np.log10(lum_dist) + 25.

    # Loading the values of the nuisance parameters for the current chain step
    alpha = lkl_input['alpha']
    beta = lkl_input['beta']
    M = lkl_input['M']
    Delta_M = lkl_input['Delta_M']

    # Computing the difference of data distance moduli - model distance moduli
    diff_dist = (self.lc_params['mb'] - (M - alpha * self.lc_params['x1']
                 + beta * self.lc_params['color'] + Delta_M *
                 (self.lc_params['3rdvar'] > self.scriptmcut)) - dist_mod)

    # Computing the cov mat with nuisance parameters using numexpr for rapidity
    cov_mat = numexpr.evaluate(
        "(mag_covmat + alpha**2 * stretch_covmat + beta**2 * colour_covmat"
        "+ 2. * alpha * mag_stretch_covmat - 2. * beta * mag_colour_covmat"
        "- 2. * alpha * beta * stretch_colour_covmat)",
        local_dict={
          'mag_covmat':self.mag_covmat,
          'stretch_covmat':self.stretch_covmat,
          'colour_covmat':self.colour_covmat,
          'mag_stretch_covmat':self.mag_stretch_covmat,
          'mag_colour_covmat':self.mag_colour_covmat,
          'stretch_colour_covmat':self.stretch_colour_covmat,
          'alpha':alpha,
          'beta':beta,
        }
    )

    # Add statistical errors to diagonal terms of the cov mat
    cov_mat += (np.diag(self.lc_params['dmb']**2 + (alpha * self.lc_params['dx1'])**2
                + (beta * self.lc_params['dcolor'])**2
                + 2. * alpha * self.lc_params['cov_m_s']
                - 2. * beta * self.lc_params['cov_m_c']
                - 2. * alpha * beta * self.lc_params['cov_s_c']))

    # Whitening the diff_dist :
    # 1)Computing the Cholesky decomposition
    cov_mat = cholesky(cov_mat, lower=True, overwrite_a=True)
    # 2)Solving the triangular system
    diff_dist  = solve_triangular(cov_mat, diff_dist, lower=True, check_finite=False)

    # Return log like
    return -0.5 * (diff_dist**2).sum()
