import os
import clik
import numpy as np

### Planck 2018 low ells TT
class likelihood:
  def __init__(self, lkl_input):
    # Some important variables
    clik_root = os.environ.get('PLANCK_PR3_DATA')
    if clik_root == None:
      raise ValueError('The environment variable PLANCK_PR3_DATA is not set.')
    self.like = clik.clik(clik_root + '/low_l/commander/commander_dx12_v3_2_29.clik')
    self.like_pars = self.like.get_extra_parameter_names()

  def get_loglike(self, class_input, lkl_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['tt'][:30] * 1e12 * class_run.T_cmb()**2.,
        np.array([lkl_input[par] for par in self.like_pars])
    ))
    return self.like(args)
