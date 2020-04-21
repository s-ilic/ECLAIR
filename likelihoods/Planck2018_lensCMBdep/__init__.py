import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_2018_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_2018_DATA is not set.')

### Planck 2018 lensing CMB-dependent
lens_CMBdep = clik.clik_lensing(clik_root + '/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing')
lens_CMBdep_pars = lens_CMBdep.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['pp'][:2501],
        class_run.lensed_cl()['tt'][:2501] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['ee'][:2501] * 1e12 * class_run.T_cmb()**2.,
        class_run.lensed_cl()['te'][:2501] * 1e12 * class_run.T_cmb()**2.,
        np.array([likes_input[par] for par in lens_CMBdep_pars])
    ))
    return lens_CMBdep(args)
