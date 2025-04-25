import numpy as np

# DESI DR2 BAO likelihood for LRG+ELG_LOPnotqso

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([0.934, 0.934])
    self.data = np.array([
       21.57563956, # DM_over_rs
       17.64149464, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [2.61732816e-02, -1.12938006e-02],
            [-1.12938006e-02, 4.04183878e-02],
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
