import os
import clik
import numpy as np

### Some important variables ###
clik_root = os.environ.get('PLANCK_PR3_DATA')
if clik_root == None:
    raise ValueError('The environment variable PLANCK_PR3_DATA is not set.')

### Planck 2018 lensing CMB-marginalized
lens_CMBmarg = clik.clik_lensing(clik_root + '/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing')
lens_CMBmarg_pars = lens_CMBmarg.get_extra_parameter_names()
def get_loglike(class_input, likes_input, class_run):
    args = np.concatenate((
        class_run.lensed_cl()['pp'][:2501],
        np.array([likes_input[par] for par in lens_CMBmarg_pars])
    ))
    return lens_CMBmarg(args)
