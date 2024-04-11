import pyactlike
import numpy as np

### ACTPol lite DR4 likelihood TT-only
class likelihood:
  def __init__(self, lkl_input):
    # Some important variables
    use_tt = True
    use_te = False
    use_ee = False
    tt_lmax = 5000
    bmin = 0
    self.like = pyactlike.ACTPowerSpectrumData(
        use_tt=use_tt,
        use_te=use_te,
        use_ee=use_ee,
        tt_lmax=tt_lmax,
        bmin=bmin,
    )

  def get_loglike(self, class_input, lkl_input, class_run):
    ell = class_run.lensed_cl()['ell'][2:]
    f = ell * (ell + 1.) / 2. / np.pi
    dell_tt = f * class_run.lensed_cl()['tt'][2:] * 1e12 * class_run.T_cmb()**2.
    dell_te = f * 0.
    dell_ee = f * 0.
    return self.like.loglike(dell_tt, dell_te, dell_ee, lkl_input['yp2'])
