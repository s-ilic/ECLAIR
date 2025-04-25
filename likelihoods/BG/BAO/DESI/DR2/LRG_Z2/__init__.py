import numpy as np

# DESI DR2 BAO likelihood for LRG in 0.6 < z < 0.8.

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([0.706, 0.706])
    self.data = np.array([
       17.35069094, # DM_over_rs
       19.45534918, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [3.23752442e-02, -2.37445646e-02],
            [-2.37445646e-02, 1.11469198e-01],
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
