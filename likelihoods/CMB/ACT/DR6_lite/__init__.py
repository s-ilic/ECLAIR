import os, sys
import numpy as np
import sacc

### ACT DR6 lite likelihood, full TTTEEE

class likelihood:
    def __init__(self, lkl_input):

        if "ACT_DR6_lite_ell_min_TT" not in lkl_input:
            ell_min_TT = 600
        else:
            ell_min_TT = lkl_input["ACT_DR6_lite_ell_min_TT"]
        if "ACT_DR6_lite_ell_min_TE" not in lkl_input:
            ell_min_TE = 600
        else:
            ell_min_TE = lkl_input["ACT_DR6_lite_ell_min_TE"]
        if "ACT_DR6_lite_ell_min_EE" not in lkl_input:
            ell_min_EE = 600
        else:
            ell_min_EE = lkl_input["ACT_DR6_lite_ell_min_EE"]
        if "ACT_DR6_lite_ell_max_TT" not in lkl_input:
            ell_max_TT = 6500
        else:
            ell_max_TT = lkl_input["ACT_DR6_lite_ell_max_TT"]
        if "ACT_DR6_lite_ell_max_TE" not in lkl_input:
            ell_max_TE = 6500
        else:
            ell_max_TE = lkl_input["ACT_DR6_lite_ell_max_TE"]
        if "ACT_DR6_lite_ell_max_EE" not in lkl_input:
            ell_max_EE = 6500
        else:
            ell_max_EE = lkl_input["ACT_DR6_lite_ell_max_EE"]

        self.ell_cuts = {
            "TT": [ell_min_TT, ell_max_TT],
            "TE": [ell_min_TE, ell_max_TE],
            "EE": [ell_min_EE, ell_max_EE]
        }

        self.data_folder = os.path.dirname(os.path.realpath(sys.argv[0]))
        input_file = sacc.Sacc.load_fits(self.data_folder +
            "/likelihoods/CMB/ACT/DR6_lite/data/dr6_data_cmbonly.fits")

        pol_dt = {"t": "0", "e": "e", "b": "b"}

        self.spec_meta = []
        self.cull = []
        idx_max = 0

        for pol in ["TT", "TE", "EE"]:
            p1, p2 = pol.lower()
            t1, t2 = pol_dt[p1], pol_dt[p2]
            dt = f"cl_{t1}{t2}"

            tracers = input_file.get_tracer_combinations(dt)

            for tr1, tr2 in tracers:
                lmin, lmax = self.ell_cuts.get(pol, (np.inf, -np.inf))
                ls, mu, ind = input_file.get_ell_cl(dt, tr1, tr2,
                                                    return_ind=True)
                mask = np.logical_and(ls >= lmin, ls <= lmax)

                if not np.all(mask):
                    self.cull.append(ind[~mask])

                if np.any(mask):
                    window = input_file.get_bandpower_windows(ind[mask])

                    self.spec_meta.append({
                        "data_type": dt,
                        "tracer1": tr1,
                        "tracer2": tr2,
                        "pol": pol.lower(),
                        "ell": ls[mask],
                        "spec": mu[mask],
                        "idx": ind[mask],
                        "window": window
                    })

                    idx_max = max(idx_max, max(ind))

        self.data_vec = np.zeros((idx_max+1,))
        for m in self.spec_meta:
            self.data_vec[m["idx"]] = m["spec"]

        self.covmat = input_file.covariance.covmat
        # OLD WAY OF CULLING THE COVMAT
        # for culls in self.cull:
        #     self.covmat[culls, :] = 0.0
        #     self.covmat[:, culls] = 0.0
        #     self.covmat[culls, culls] = 1e10

        # SILIC: NEW WAY OF CULLING THE COVMAT
        self.keep_idx = []
        for i in range(self.covmat.shape[0]):
            if all([i not in culls for culls in self.cull]):
                self.keep_idx.append(i)
        self.covmat = self.covmat[:, self.keep_idx][self.keep_idx, :]

        self.inv_cov = np.linalg.inv(self.covmat)

    def chi_square(self, cl, A_act, P_act):
        ps_vec = np.zeros_like(self.data_vec)

        for m in self.spec_meta:
            idx = m["idx"]
            win = m["window"].weight.T
            ls = m["window"].values
            pol = m["pol"]
            dat = cl[pol][ls] / (A_act * A_act)
            if pol[0] == "e":
                dat /= P_act
            if pol[1] == "e":
                dat /= P_act

            ps_vec[idx] = win @ dat

        delta = self.data_vec - ps_vec
        delta = delta[self.keep_idx]

        chisquare = delta @ self.inv_cov @ delta

        return chisquare

    def loglike(self, cl, A_act, P_act):
        return -0.5 * self.chi_square(cl, A_act, P_act)

    # def logp(self, **param_values):
    def get_loglike(self, class_input, lkl_input, class_run):

        T_CMB = class_run.T_cmb()

        # Recover Cl_s from CLASS
        cl_theo = {}
        fl = class_run.lensed_cl()['ell'] * (class_run.lensed_cl()['ell'] + 1.) / (2. * np.pi)
        cl_theo["tt"] = fl * class_run.lensed_cl()['tt'] * 1e12 * T_CMB**2.
        cl_theo["te"] = fl * class_run.lensed_cl()['te'] * 1e12 * T_CMB**2.
        cl_theo["ee"] = fl * class_run.lensed_cl()['ee'] * 1e12 * T_CMB**2.

        return self.loglike(cl_theo, lkl_input["A_act"], lkl_input["P_act"])