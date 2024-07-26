import os, sys
import numpy as np

# Likelihood for the Union3 & UNITY1.5 type Ia supernovae sample.
# from https://arxiv.org/abs/2311.12098

class likelihood:
    def __init__(self, lkl_input):
        ### Path to Union3 likelihood folder (i.e. the folder this file is in)
        path_to_data = os.path.dirname(os.path.realpath(sys.argv[0]))
        path_to_data += '/likelihoods/BG/SN/Union3/'
        data_file = path_to_data + "lcparam_full.txt"
        covmat_file =  path_to_data + "mag_covmat.txt"
        self.twoscriptmfit = False
        self._read_data_file(data_file)
        self.covs = {}
        for name in ['mag']:
            self.covs[name] = self._read_covmat(covmat_file)
        self.alphabeta_covmat = False
        self.configure()
        self.inverse_covariance_matrix()
        self.use_abs_mag = False
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False

    def _apply_mask(self, zmask):
        for col in self.cols:
            setattr(self, col, getattr(self, col)[zmask])
        for name, cov in self.covs.items():
            self.covs[name] = cov[np.ix_(zmask, zmask)]

    def configure(self):
        self.pre_vars = 0.  # diagonal component

    def _read_covmat(self, filename):
        cov = np.loadtxt(filename)
        if np.isscalar(cov[0]) and cov[0] ** 2 + 1 == len(cov):
            cov = cov[1:]
        return cov.reshape((self.nsn, self.nsn))

    def _read_cols(self, data_file, file_cols, sep=None):
        with open(data_file, 'r') as f:
            lines = f.readlines()
            line = lines[0]
            if line.startswith('#'):
                line = line[1:]
            cols = [col.strip().lower() for col in line.split(sep)]
            assert cols[0].isalpha()
            indices = [cols.index(col) for col in file_cols]
            zeros = np.zeros(len(lines) - 1)
            for col in self.cols:
                setattr(self, col, zeros.astype(dtype='f8', copy=True))
            for ix, line in enumerate(lines[1:]):
                vals = [val.strip() for val in line.split(sep)]
                vals = [vals[i] for i in indices]
                for i, (col, val) in enumerate(zip(self.cols, vals)):
                    tmp = getattr(self, col)
                    tmp[ix] = np.asarray(val, dtype=tmp.dtype)
        self.nsn = ix + 1

    def _read_data_file(self, data_file):
        file_cols = ['zcmb', 'mb']
        self.cols = ['zcmb', 'mag']
        self._read_cols(data_file, file_cols)
        self.zhel = self.zcmb

    def _marginalize_abs_mag(self):
        deriv = np.ones_like(self.mag)[:, None]
        derivp = self.invcov.dot(deriv)
        fisher = deriv.T.dot(derivp)
        self.invcov = self.invcov - derivp.dot(np.linalg.solve(fisher, derivp.T))

    def inverse_covariance_matrix(self, alpha=0, beta=0):
        if 'mag' in self.covs:
            invcovmat = self.covs['mag'].copy()
        else:
            invcovmat = 0
        if self.alphabeta_covmat:
            if np.isclose(alpha, self._last_alpha) and np.isclose(beta, self._last_beta):
                return self.invcov
            self._last_alpha = alpha
            self._last_beta = beta
            alphasq = alpha * alpha
            betasq = beta * beta
            alphabeta = alpha * beta
            if 'stretch' in self.covs:
                invcovmat += alphasq * self.covs['stretch']
            if 'colour' in self.covs:
                invcovmat += betasq * self.covs['colour']
            if 'mag_stretch' in self.covs:
                invcovmat += 2 * alpha * self.covs['mag_stretch']
            if 'mag_colour' in self.covs:
                invcovmat -= 2 * beta * self.covs['mag_colour']
            if 'stretch_colour' in self.covs:
                invcovmat -= 2 * alphabeta * self.covs['stretch_colour']
            delta = (self.pre_vars + alphasq * self.stretch_var +
                     betasq * self.colour_var + 2.0 * alpha * self.cov_mag_stretch +
                     -2.0 * beta * self.cov_mag_colour +
                     -2.0 * alphabeta * self.cov_stretch_colour)
        else:
            delta = self.pre_vars
        np.fill_diagonal(invcovmat, invcovmat.diagonal() + delta)
        self.invcov = np.linalg.inv(invcovmat)
        return self.invcov

    def alpha_beta_logp(self, lumdists, Mb=0., **kwargs):
        if self.use_abs_mag:
            estimated_scriptm = Mb + 25
        else:
            estimated_scriptm = 0.
        diffmag = self.mag - lumdists - estimated_scriptm
        return - diffmag.dot(self.invcov).dot(diffmag) / 2.

    def get_loglike(self, class_input, lkl_input, class_run):
        angular_diameter_distances = [class_run.angular_distance(z) for z in self.zcmb]
        lumdists = (5 * np.log10((1 + self.zhel) * (1 + self.zcmb) *
                                 np.array(angular_diameter_distances)))

        if self.use_abs_mag:
            Mb = lkl_input['Mb']
        else:
            Mb = 0
        if self.marginalize:
            # Should parallelize this loop
            for i in range(self.int_points):
                self.marge_grid[i] = - self.alpha_beta_logp(
                    lumdists, self.alpha_grid[i],
                    self.beta_grid[i], Mb,
                    invcovmat=self.invcovs[i])
            grid_best = np.min(self.marge_grid)
            return - grid_best + np.log(
                np.sum(np.exp(- self.marge_grid[self.marge_grid != np.inf] + grid_best)) *
                self.step_width_alpha * self.step_width_beta)
        else:
            if self.alphabeta_covmat:
                return self.alpha_beta_logp(lumdists, lkl_input[self.alpha_name],
                                            lkl_input[self.beta_name], Mb)
            else:
                return self.alpha_beta_logp(lumdists, Mb=Mb)