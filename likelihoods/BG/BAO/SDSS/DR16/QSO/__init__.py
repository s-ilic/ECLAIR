import numpy as np

# BAO eBOSS DR16 0.6<z<1.0
# Consensus (Fourier and Configuration space) results by eBOSS QSO team:
# Jiamin , Richard
# https://arxiv.org/abs/2007.XXYY
# https://arxiv.org/abs/2007.XXYY

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([1.48, 1.48])
    self.data = np.array([
       30.6876, # DM_over_rs
       13.2609, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [0.63731604, 0.1706891],
            [0.1706891, 0.30468415],
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
