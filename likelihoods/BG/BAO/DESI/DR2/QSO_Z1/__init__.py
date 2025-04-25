import numpy as np

# DESI DR2 BAO likelihood for QSO

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([1.484, 1.484])
    self.data = np.array([
       30.51190063, # DM_over_rs
       12.81699964, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [5.83020277e-01, -1.95215562e-01],
            [-1.95215562e-01, 2.68336193e-01],
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