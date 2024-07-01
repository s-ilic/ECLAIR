import os, sys
import act_dr6_lenslike as alike

path_to_data = os.path.dirname(os.path.realpath(sys.argv[0]))
path_to_data += '/likelihoods/CMB/ACT/DR6_lensing/data/'

### ACT DR6 lensing likelihood
class likelihood:
  def __init__(self, lkl_input):
    variant = 'act_baseline'
    lens_only = False
    like_corrections = True
    self.data_dict = alike.load_data(
      variant,
      lens_only=lens_only,
      like_corrections=like_corrections,
      ddir=path_to_data
    )

  def get_loglike(self, class_input, lkl_input, class_run):
    ell_kk = class_run.lensed_cl()['ell']
    ell_cmb = class_run.lensed_cl()['ell']
    cl_tt = class_run.lensed_cl()['tt'] * 1e12 * class_run.T_cmb()**2.
    cl_te = class_run.lensed_cl()['te'] * 1e12 * class_run.T_cmb()**2.
    cl_ee = class_run.lensed_cl()['ee'] * 1e12 * class_run.T_cmb()**2.
    cl_bb = class_run.lensed_cl()['bb'] * 1e12 * class_run.T_cmb()**2.
    cl_kk = 0.25*(ell_kk*(ell_kk+1))**2.*class_run.lensed_cl()['pp']
    lnlike = alike.generic_lnlike(
      self.data_dict,
      ell_kk,
      cl_kk,
      ell_cmb,
      cl_tt,
      cl_ee,
      cl_te,
      cl_bb,
    )
    return lnlike
