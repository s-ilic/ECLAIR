import numpy as np

# BAO likelihoods corresponding to the arXiv release on April 5, 2024, see:
# - https://arxiv.org/abs/2404.03000
# - https://arxiv.org/abs/2404.03001
# - https://arxiv.org/abs/2404.03002
# Here: LRG+ELG, 0.8 < z < 1.1

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([0.930, 0.930])
    self.data = np.array([
       21.70841761, # DM_over_rs
       17.87612922, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [7.95675235e-02, -3.80110101e-02],
            [-3.80110101e-02, 1.19935683e-01],
        ]
    )
    self.icov_mat = np.linalg.inv(cov_mat)

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag()
    theo = np.zeros(2)
    DMz = class_run.angular_distance(self.z[0]) * (1. + self.z[0])
    DHz = 1. / class_run.Hubble(self.z[1])
    theo[0] = DMz / rs
    theo[1] = DHz / rs
    diff = self.data - theo
    lnl = -0.5 * np.dot(np.dot(diff, self.icov_mat), diff)
    return lnl
