import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_PR2_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_PR2_DATA is not set.')

### Planck 2015 low ells TT
lell_TT = clik.clik(clik_root + '/low_l/commander/commander_rc2_v1.1_l2_29_B.clik')
lell_TT_pars = lell_TT.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['tt'][:30] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in lell_TT_pars])
    ))
    return lell_TT(args)
