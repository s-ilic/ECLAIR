import os
import clik
import numpy as np

### Planck 2018 lensing CMB-marginalized
class likelihood:
  def __init__(self, lkl_input):
    # Some important variables
    clik_root = os.environ.get('PLANCK_PR3_DATA')
    if clik_root == None:
      raise ValueError('The environment variable PLANCK_PR3_DATA is not set.')
    self.like = clik.clik_lensing(clik_root + '/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing')
    self.like_pars = self.like.get_extra_parameter_names()

  def get_loglike(self, class_input, lkl_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['pp'][:2501],
        np.array([lkl_input[par] for par in self.like_pars])
    ))
    return self.like(args)[0]
