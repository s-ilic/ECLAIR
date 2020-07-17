import pyactlike

### Some important variables ###
use_tt = False
use_te = True
use_ee = False
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
    dell_tt = class_run.lensed_cl()['tt'][2:tt_lmax+1] * 1e12 * class_run.T_cmb()**2.
    dell_te = class_run.lensed_cl()['te'][2:tt_lmax+1] * 1e12 * class_run.T_cmb()**2.
    dell_ee = class_run.lensed_cl()['ee'][2:tt_lmax+1] * 1e12 * class_run.T_cmb()**2.
    return like.loglike(dell_tt, dell_te, dell_ee, likes_input['yp2'])
