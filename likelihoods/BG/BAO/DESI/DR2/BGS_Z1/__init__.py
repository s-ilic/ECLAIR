import numpy as np

# DESI DR2 BAO likelihood for BGS_BRIGHT-21.35

class likelihood:
  def __init__(self, lkl_input):
    self.z = 0.295
    self.data = 7.94167639 # DV_over_rs
    self.cov = 5.78998687e-03

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag()
    theo = np.cbrt(((1.+self.z)*class_run.angular_distance(self.z))**2.
                   * self.z / class_run.Hubble(self.z)) / rs
    diff = self.data - theo
    lnl = -0.5 * diff**2. / self.cov
    return lnl
