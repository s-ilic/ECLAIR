import os
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline

# DR16 Lya Auto

class likelihood:
  def __init__(self, lkl_input):
    self.z = 2.334
    path_to_data = os.path.dirname(os.path.realpath(sys.argv[0]))
    path_to_data += '/likelihoods/BG/BAO/SDSS/DR16/LYAUTO/'
    data = np.loadtxt(path_to_data + "sdss_DR16_LYAUTO_BAO_DMDHgrid.txt")
    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    self.x_min, self.x_max = data[:, 0].min(), data[:, 0].max()
    self.y_min, self.y_max = data[:, 1].min(), data[:, 1].max()
    Nx = x.shape[0]
    Ny = y.shape[0]
    z = np.reshape(data[:, 2], [Nx, Ny])
    self.logprob = RectBivariateSpline(x, y, np.log(z), kx=3, ky=3)

  def check_bounds(self, x, y):
    return (self.x_min <= x <= self.x_max) & (self.y_min <= y <= self.y_max)

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag()
    DMz = class_run.angular_distance(self.z) * (1. + self.z)
    DHz = 1. / class_run.Hubble(self.z)
    if self.check_bounds(DMz / rs, DHz / rs):
      lnl = self.logprob(DMz / rs, DHz / rs)
      return lnl[0][0]
    else:
      return -np.inf
