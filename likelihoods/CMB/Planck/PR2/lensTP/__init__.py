import os
import clik
import numpy as np

### Planck 2015 lensing full T+P
class likelihood:
  def __init__(self, lkl_input):
    # Some important variables
    clik_root = os.environ.get('PLANCK_PR2_DATA')
    if clik_root == None:
      raise ValueError('The environment variable PLANCK_PR2_DATA is not set.')
    self.like = clik.clik_lensing(clik_root + '/lensing/smica_g30_ftl_full_pp.clik_lensing')
    self.like_pars = self.like.get_extra_parameter_names()

  def get_loglike(self, class_input, lkl_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['pp'][:2049],
        class_run.lensed_cl()['tt'][:2049] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['ee'][:2049] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['te'][:2049] * 1e12 * class_run.T_cmb()**2.,
        np.array([lkl_input[par] for par in self.like_pars])
    ))
    return self.like(args)[0]
