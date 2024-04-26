import os
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline

# DR16 ELG

class likelihood:
  def __init__(self, lkl_input):
    self.zeff = 0.845
    path_to_data = os.path.dirname(os.path.realpath(sys.argv[0]))
    path_to_data += '/likelihoods/BG/BAO/SDSS/DR16/ELG/'
    data = np.loadtxt(path_to_data + "sdss_DR16_ELG_BAO_DVtable.txt")
    self.prob = UnivariateSpline(data[:, 0], data[:, 1], s=0, ext=2)

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag()
    theo = np.cbrt(((1.+self.zeff)*class_run.angular_distance(self.zeff))**2.
                   * self.zeff / class_run.Hubble(self.zeff)) / rs
    try:
        lnl = np.log(self.prob(theo))
        return lnl
    except:
       return -np.inf
