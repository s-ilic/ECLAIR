import os
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline

# DR16 ELG

class likelihood:
  def __init__(self, lkl_input):
    self.zeff = 0.15
    #self.data_DV_over_rs = 4.465666824
    #self.error_DV_over_rs = 0.1681350461
    self.DV_fid = 638.9518
    self.rs_fid = 148.69
    path_to_data = os.path.dirname(os.path.realpath(sys.argv[0]))
    path_to_data += '/likelihoods/BG/BAO/SDSS/DR7/MGS/'
    data = np.loadtxt(path_to_data + "chid_MGSconsensus.dat")
    self.chi2 = UnivariateSpline(data[:, 0], data[:, 1], s=0, ext=2)

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag()
    theo = np.cbrt(((1.+self.zeff)*class_run.angular_distance(self.zeff))**2.
                   * self.zeff / class_run.Hubble(self.zeff)) / rs
    theo_alpha = theo * self.rs_fid / self.DV_fid
    try:
        lnl = -0.5 * self.chi2(theo_alpha)
        return lnl
    except:
       return -np.inf
