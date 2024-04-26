import numpy as np

# BAO eBOSS DR16 0.6<z<1.0
# Consensus (Fourier and Configuration space) results by eBOSS LRG team:
# Julian Bautista, Hector Gil-Marin, Mariana Varga Magana, Romain Paviot,
# Sylvain de la Torre
# https://arxiv.org/abs/2007.XXYY

class likelihood:
  def __init__(self, lkl_input):
    self.z = np.array([0.698, 0.698])
    self.data = np.array([
       17.85823691865007, # DM_over_rs
       19.32575373059217, # DH_over_rs
    ])
    cov_mat = np.array(
        [
            [0.1076634008565565, -0.05831820341302727],
            [-0.05831820341302727, 0.2838176386340292],
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
