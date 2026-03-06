import numpy as np
from copy import deepcopy
import candl
import spt_candl_data

### SPT3G D1 full TTTEEE likelihood

class likelihood:
    def __init__(self, lkl_input):

        self.candl_like = candl.Like(spt_candl_data.SPT3G_D1_TnE, variant="lite")

    def get_loglike(self, class_input, lkl_input, class_run):

        # Recover nuisance parameters
        pars_for_like = deepcopy(lkl_input)

        # Recover relevant quantities from CLASS
        cl_theo = {}
        T_CMB = class_run.T_cmb()
        i_max = self.candl_like.ell_max + 1
        cl_theo["ell"] = class_run.lensed_cl()['ell'][2:i_max]
        fl = cl_theo["ell"] * (cl_theo["ell"] + 1.) / (2. * np.pi)
        cl_theo["TT"] = fl * class_run.lensed_cl()['tt'][2:i_max] * 1e12 * T_CMB**2.
        cl_theo["TE"] = fl * class_run.lensed_cl()['te'][2:i_max] * 1e12 * T_CMB**2.
        cl_theo["EE"] = fl * class_run.lensed_cl()['ee'][2:i_max] * 1e12 * T_CMB**2.
        if "bb" in class_run.lensed_cl():
            cl_theo["BB"] = fl * class_run.lensed_cl()['bb'][2:i_max] * 1e12 * T_CMB**2.
        if "pp" in class_run.lensed_cl():
            cl_theo["pp"] = (class_run.lensed_cl()['pp'][2:i_max] * ((cl_theo["ell"] * (cl_theo["ell"] + 1)) ** 2.0) / (2.0 * np.pi))
            cl_theo["kk"] = cl_theo["pp"] * np.pi / 2.0
        pars_for_like["Dl"] = deepcopy(cl_theo)

        # Add tau to the parameters for the likelihood
        pars_for_like["tau"] = class_run.tau_reio()

        # Compute the likelihood
        return self.candl_like.log_like(pars_for_like)