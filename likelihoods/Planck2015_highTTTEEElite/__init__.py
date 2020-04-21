import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_2015_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_2015_DATA is not set.')

### Planck 2015 lite high ells TT, TE, EE
hell_TTTEEElite = clik.clik(clik_root + '/hi_l/plik_lite/plik_lite_v18_TTTEEE.clik')
hell_TTTEEElite_pars = hell_TTTEEElite.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['tt'][:2509] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['ee'][:2509] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['te'][:2509] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in hell_TTTEEElite_pars])
    ))
    return hell_TTTEEElite(args)
