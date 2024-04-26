import numpy as np

# BAO BOSS DR12 LRG 0.2<z<0.5 and 0.4<z<0.6
# Based on Alam et al. 2016
# https://arxiv.org/abs/1607.03155

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([0.38, 0.38, 0.51, 0.51])
    self.data = np.array([
       1.023406e+01, # DM_over_rs
       2.498058e+01, # DH_over_rs
       1.336595e+01, # DM_over_rs
       2.231656e+01, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [2.860520e-02, -4.939281e-02, 1.489688e-02, -1.387079e-02],
            [-4.939281e-02, 5.307187e-01, -2.423513e-02, 1.767087e-01],
            [1.489688e-02, -2.423513e-02, 4.147534e-02, -4.873962e-02],
            [-1.387079e-02, 1.767087e-01, -4.873962e-02, 3.268589e-01],
        ]
    )
    self.icov_mat = np.linalg.inv(cov_mat)

  def get_loglike(self, class_input, lkl_input, class_run):
    rs = class_run.rs_drag()
    theo = np.zeros(4)
    for i in [0, 2]:
        DMz = class_run.angular_distance(self.z[i]) * (1. + self.z[i])
        DHz = 1. / class_run.Hubble(self.z[i+1])
        theo[i] = DMz / rs
        theo[i+1] = DHz / rs
    diff = self.data - theo
    lnl = -0.5 * np.dot(np.dot(diff, self.icov_mat), diff)
    return lnl
