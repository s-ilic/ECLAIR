import numpy as np

### From Freedman et al., 2002.01550 
class likelihood:
  def __init__(self, lkl_input):
    self.H0_data = 69.6
    self.H0_data_sigma = 2.5

  def get_loglike(self, class_input, lkl_input, class_run):
    return -0.5 * (class_run.h() * 100 - self.H0_data)**2. / self.H0_data_sigma**2.
