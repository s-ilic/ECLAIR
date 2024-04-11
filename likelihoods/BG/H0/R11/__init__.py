import numpy as np

### From Riess et al., 1103.2976
class likelihood:
  def __init__(self, lkl_input):
    self.H0_data = 73.8
    self.H0_data_sigma = 2.4

  def get_loglike(self, class_input, lkl_input, class_run):
    return -0.5 * (class_run.h() * 100 - self.H0_data)**2. / self.H0_data_sigma**2.
