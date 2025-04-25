import numpy as np

# DESI DR2 BAO likelihood for Lya

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([2.33, 2.33])
    self.data = np.array([
       8.631545674846294, # DH_over_rs
       38.988973961958784, # DM_over_rs
    ])
    cov_mat = np.array(
        [
            [1.021361941704139985e-02, -2.313952164707928916e-02],
            [-2.313952164707928916e-02, 2.826857787839159308e-01],
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
