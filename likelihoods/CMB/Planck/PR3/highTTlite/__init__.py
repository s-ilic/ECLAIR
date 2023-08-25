import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_PR3_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_PR3_DATA is not set.')

### Planck 2018 lite high ells TT
hell_TTlite = clik.clik(clik_root + '/hi_l/plik_lite/plik_lite_v22_TT.clik')
hell_TTlite_pars = hell_TTlite.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['tt'][:2509] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in hell_TTlite_pars])
    ))
    return hell_TTlite(args)
