import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_2015_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_2015_DATA is not set.')

### Planck 2015 lensing full T+P
lensTP = clik.clik_lensing(clik_root + '/lensing/smica_g30_ftl_full_pp.clik_lensing')
lensTP_pars = lensTP.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['pp'][:2049],
        class_run.lensed_cl()['tt'][:2049] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['ee'][:2049] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['te'][:2049] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in lensTP_pars])
    ))
    return lensTP(args)
