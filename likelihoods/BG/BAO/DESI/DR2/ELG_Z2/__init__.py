import numpy as np

# DESI DR2 BAO likelihood for ELG_LOPnotqso in 1.1 < z < 1.6

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([1.321, 1.321])
    self.data = np.array([
       27.60085612, # DM_over_rs
       14.17602155, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [1.05336516e-01, -2.90308418e-02],
            [-2.90308418e-02, 5.04233092e-02],
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
