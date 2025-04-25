import numpy as np

# BAO likelihoods corresponding to the arXiv release on April 5, 2024, see:
# - https://arxiv.org/abs/2404.03000
# - https://arxiv.org/abs/2404.03001
# - https://arxiv.org/abs/2404.03002
# Here: QSO, 0.8 < z < 2.1

class likelihood:
  def __init__(self, lkl_input):
    self.z = 1.491
    self.data = 26.07217182 # DV_over_rs
    self.cov = 4.47134991e-01

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag()
    theo = np.cbrt(((1.+self.z)*class_run.angular_distance(self.z))**2.
                   * self.z / class_run.Hubble(self.z)) / rs
    diff = self.data - theo
    lnl = -0.5 * diff**2. / self.cov
    return lnl