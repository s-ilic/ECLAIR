import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_PR3_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_PR3_DATA is not set.')

### Planck 2018 low ells BB
lell_BB = clik.clik(clik_root + '/low_l/simall/simall_100x143_offlike5_BB_Aplanck_B.clik')
lell_BB_pars = lell_BB.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['bb'][:30] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in lell_BB_pars])
    ))
    return lell_BB(args)
