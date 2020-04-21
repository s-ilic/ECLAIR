import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_2018_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_2018_DATA is not set.')

### Planck 2018 low ells TT
lell_TT = clik.clik(clik_root + '/low_l/commander/commander_dx12_v3_2_29.clik')
lell_TT_pars = lell_TT.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['tt'][:30] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in lell_TT_pars])
    ))
    return lell_TT(args)
