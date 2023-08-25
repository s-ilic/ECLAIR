import pyactlike
import numpy as np

### Some important variables ###
use_tt = True
use_te = True
use_ee = True
tt_lmax = 5000
bmin = 0
like = pyactlike.ACTPowerSpectrumData(
    use_tt=use_tt,
    use_te=use_te,
    use_ee=use_ee,
    tt_lmax=tt_lmax,
    bmin=bmin,
)

### ACTPol lite DR4 likelihood
def get_loglike(class_input, likes_input, class_run):
    ell = class_run.lensed_cl()['ell'][2:]
    f = ell * (ell + 1.) / 2. / np.pi
    dell_tt = f * class_run.lensed_cl()['tt'][2:] * 1e12 * class_run.T_cmb()**2.
    dell_te = f * class_run.lensed_cl()['te'][2:] * 1e12 * class_run.T_cmb()**2.
    dell_ee = f * class_run.lensed_cl()['ee'][2:] * 1e12 * class_run.T_cmb()**2.
    return like.loglike(dell_tt, dell_te, dell_ee, likes_input['yp2'])
