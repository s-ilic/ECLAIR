import numpy as np

# BAO likelihoods corresponding to the arXiv release on April 5, 2024, see:
# - https://arxiv.org/abs/2404.03000
# - https://arxiv.org/abs/2404.03001
# - https://arxiv.org/abs/2404.03002
# Here: Lya z ~ 2.33

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([2.33, 2.33])
    self.data = np.array([
       8.522565833956225, # DH_over_rs
       39.70838281345702, # DM_over_rs
    ])
    cov_mat = np.array(
        [
            [2.918604472134976591e-02, -7.694771196027801186e-02],
            [-7.694771196027801186e-02, 8.897529278758126159e-01],
        ]
    )
    self.icov_mat = np.linalg.inv(cov_mat)

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag()
    theo = np.zeros(2)
    DHz = 1. / class_run.Hubble(self.z[0])
    DMz = class_run.angular_distance(self.z[1]) * (1. + self.z[1])
    theo[0] = DHz / rs
    theo[1] = DMz / rs
    diff = self.data - theo
    lnl = -0.5 * np.dot(np.dot(diff, self.icov_mat), diff)
    return lnl
