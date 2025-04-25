import numpy as np

# BAO likelihoods corresponding to the arXiv release on April 5, 2024, see:
# - https://arxiv.org/abs/2404.03000
# - https://arxiv.org/abs/2404.03001
# - https://arxiv.org/abs/2404.03002
# Here: LRG, 0.6 < z < 0.8

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([0.706, 0.706])
    self.data = np.array([
       16.84645313, # DM_over_rs
       20.07872919, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [1.01975713e-01, -7.99403059e-02],
            [-7.99403059e-02, 3.54449156e-01],
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
