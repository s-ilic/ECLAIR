import numpy as np

# Likelihood of the BAO detection of the 6dF Galaxy Survey
# from Beutler et al. 2011, https://arxiv.org/abs/1106.3366

class likelihood:
  def __init__(self, lkl_input):
    self.z = 0.106
    self.data = 0.336 # rs_over_DV
    self.stddev = 0.015
    self.rs_rescale = 1.027369826

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag() * self.rs_rescale
    theo = rs / np.cbrt(((1.+self.z)*class_run.angular_distance(self.z))**2.
                        * self.z / class_run.Hubble(self.z))
    diff = self.data - theo
    lnl = -0.5 * diff**2. / self.stddev**2.
    return lnl