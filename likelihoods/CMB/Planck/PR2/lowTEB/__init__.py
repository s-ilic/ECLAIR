import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_2015_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_2015_DATA is not set.')

### Planck 2015 low ells TT, EE, BB
lell_TEB = clik.clik(clik_root + '/low_l/bflike/lowl_SMW_70_dx11d_2014_10_03_v5c_Ap.clik')
lell_TEB_pars = lell_TEB.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['tt'][:30] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['ee'][:30] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['bb'][:30] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['te'][:30] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in lell_TEB_pars])
    ))
    return lell_TEB(args)
