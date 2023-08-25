import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_PR3_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_PR3_DATA is not set.')

### Planck 2018 high ells TT, TE, EE
hell_TTTEEE = clik.clik(clik_root + '/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik')
hell_TTTEEE_pars = hell_TTTEEE.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['tt'][:2509] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['ee'][:2509] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['te'][:2509] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in hell_TTTEEE_pars])
    ))
    return hell_TTTEEE(args)
