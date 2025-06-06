# Global
import os
import numpy as np
from typing import List
from getdist import ParamNames
from scipy.linalg import sqrtm


planck_pr4_root = os.environ.get('PLANCK_PR4_DATA')
if planck_pr4_root == None:
    raise ValueError('The environment variable PLANCK_PR4_DATA is not set.')
planck_pr4_root += "/lensing/"

class likelihood:

    # Data type for aggregated chi2 (case sensitive)
    type = "CMB"

    # used to form spectra names, e.g. AmapxBmap
    map_separator: str = 'x'

    def __init__(self, lkl_input):
        self.like_approx = "gaussian"
        self.fields_use = ["P"]
        self.fields_required =  ["P"]
        self.binned = True
        self.nbins= 9
        self.use_min= 1
        self.use_max= 9
        self.cl_lmin = 2
        self.cl_lmax = 2500

        self.cl_hat_file = planck_pr4_root + "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_bandpowers.dat"

        self.bin_window_files = planck_pr4_root + "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_window/window%u.dat"
        self.bin_window_in_order = ["PP"]
        self.bin_window_out_order = ["PP"]

        self.covmat_cl = "PP"
        self.covmat_fiducial = planck_pr4_root + "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_CMBmarged_cov.dat"

        self.linear_correction_fiducial_file = planck_pr4_root + "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_CMBmarged_lensing_fiducial_correction.dat"
        self.linear_correction_bin_window_files = planck_pr4_root + "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_CMBmarged_lens_delta_window/window%u.dat"
        self.linear_correction_bin_window_in_order = ["PP"]
        self.linear_correction_bin_window_out_order =  ["PP"]

        self.calibration_param = None

        ####################################################################""

        self.field_names = ['T', 'E', 'B', 'P']
        self.tot_theory_fields = len(self.field_names)
        self.map_names = []
        self.has_map_names = bool(self.map_names)
        if self.has_map_names:
            # e.g. have multiple frequencies for given field measurement
            #map_fields = ini.split('map_fields')
            #if len(map_fields) != len(self.map_names):
            #    raise ValueError('number of map_fields does not match map_names')
            #self.map_fields = [self.typeIndex(f) for f in map_fields]
            pass
        else:
            self.map_names = self.field_names
            self.map_fields = np.arange(len(self.map_names), dtype=int)
        fields_use = self.fields_use
        if len(fields_use):
            index_use = [self.typeIndex(f) for f in fields_use]
            use_theory_field = [i in index_use for i in range(self.tot_theory_fields)]
        else:
            if not self.has_map_names:
                raise ValueError('must have fields_use or map_names')
            use_theory_field = [True] * self.tot_theory_fields
        maps_use = []
        if len(maps_use):
            if any(not i for i in use_theory_field):
                print('maps_use overrides fields_use')
            self.use_map = [False] * len(self.map_names)
            for j, map_used in enumerate(maps_use):
                if map_used in self.map_names:
                    self.use_map[self.map_names.index(map_used)] = True
                else:
                    raise ValueError('maps_use item not found - %s' % map_used)
        else:
            self.use_map = [use_theory_field[self.map_fields[i]]
                            for i in range(len(self.map_names))]
        # Bandpowers can depend on more fields than are actually used in likelihood
        # e.g. for correcting leakage or other linear corrections
        self.require_map = self.use_map[:]
        if self.has_map_names:
            if self.fields_required is not None:
                raise ValueError('use maps_required not fields_required')
            maps_use = self.maps_required
        else:
            maps_use = self.fields_required
        if len(maps_use):
            for j, map_used in enumerate(maps_use):
                if map_used in self.map_names:
                    self.require_map[self.map_names.index(map_used)] = True
                else:
                    raise ValueError('required item not found %s' % map_used)
        self.required_theory_field = [False for _ in self.field_names]
        for i in range(len(self.map_names)):
            if self.require_map[i]:
                self.required_theory_field[self.map_fields[i]] = True
        self.ncl_used = 0  # set later reading covmat
        #self.like_approx = ini.string('like_approx', 'gaussian')
        self.nmaps = np.count_nonzero(self.use_map)
        self.nmaps_required = np.count_nonzero(self.require_map)
        self.required_order = np.zeros(self.nmaps_required, dtype=int)
        self.map_required_index = -np.ones(len(self.map_names), dtype=int)
        ix = 0
        for i in range(len(self.map_names)):
            if self.require_map[i]:
                self.map_required_index[i] = ix
                self.required_order[ix] = i
                ix += 1
        self.map_used_index = -np.ones(len(self.map_names), dtype=int)
        ix = 0
        self.used_map_order = []
        for i, map_name in enumerate(self.map_names):
            if self.use_map[i]:
                self.map_used_index[i] = ix
                self.used_map_order.append(map_name)
                ix += 1
        self.ncl = (self.nmaps * (self.nmaps + 1)) // 2
        self.pcl_lmax = self.cl_lmax
        self.pcl_lmin = self.cl_lmin
        #self.binned = ini.bool('binned', True)
        if self.binned:
            #self.nbins = ini.int('nbins')
            self.bin_min = self.use_min - 1
            self.bin_max = self.use_max - 1
            # needed by read_bin_windows:
            self.nbins_used = self.bin_max - self.bin_min + 1
            self.bins = self.read_bin_windows('bin_window')
        else:
            # if self.nmaps != self.nmaps_required:
            #     raise ValueError('unbinned likelihood must have nmaps==nmaps_required')
            # self.nbins = self.pcl_lmax - self.pcl_lmin + 1
            # if self.like_approx != 'exact':
            #     print('Unbinned likelihoods untested in this version')
            # self.bin_min = ini.int('use_min', self.pcl_lmin)
            # self.bin_max = ini.int('use_max', self.pcl_lmax)
            # self.nbins_used = self.bin_max - self.bin_min + 1
            pass
        self.full_bandpower_headers, self.full_bandpowers, self.bandpowers = \
            self.read_cl_array('cl_hat', return_full=True)
        #if self.like_approx == 'HL':
        #    self.cl_fiducial = self.read_cl_array(ini, 'cl_fiducial')
        #else:
        self.cl_fiducial = None
        #includes_noise = ini.bool('cl_hat_includes_noise', False)
        self.cl_noise = None
        #if self.like_approx != 'gaussian' or includes_noise:
        #    self.cl_noise = self.read_cl_array(ini, 'cl_noise')
        #    if not includes_noise:
        #        self.bandpowers += self.cl_noise
        #    elif self.like_approx == 'gaussian':
        #        self.bandpowers -= self.cl_noise
        self.cl_lmax = np.zeros((self.tot_theory_fields, self.tot_theory_fields))
        for i in range(self.tot_theory_fields):
            if self.required_theory_field[i]:
                self.cl_lmax[i, i] = self.pcl_lmax
        if self.required_theory_field[0] and self.required_theory_field[1]:
            self.cl_lmax[1, 0] = self.pcl_lmax

        #if self.like_approx != 'gaussian':
        #    cl_fiducial_includes_noise = ini.bool('cl_fiducial_includes_noise', False)
        #else:
        cl_fiducial_includes_noise = False
        self.bandpower_matrix = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
        self.noise_matrix = self.bandpower_matrix.copy()
        self.fiducial_sqrt_matrix = self.bandpower_matrix.copy()
        if self.cl_fiducial is not None and not cl_fiducial_includes_noise:
            self.cl_fiducial += self.cl_noise
        for b in range(self.nbins_used):
            self.elements_to_matrix(self.bandpowers[:, b], self.bandpower_matrix[b, :, :])
            if self.cl_noise is not None:
                self.elements_to_matrix(self.cl_noise[:, b], self.noise_matrix[b, :, :])
            if self.cl_fiducial is not None:
                self.elements_to_matrix(self.cl_fiducial[:, b],
                                        self.fiducial_sqrt_matrix[b, :, :])
                self.fiducial_sqrt_matrix[b, :, :] = (
                    sqrtm(self.fiducial_sqrt_matrix[b, :, :]))
        if self.like_approx == 'exact':
            #self.fsky = ini.float('fullsky_exact_fksy')
            pass
        else:
            self.cov = self.ReadCovmat()
            self.covinv = np.linalg.inv(self.cov)
        if self.linear_correction_fiducial_file is not None:
            self.fid_correction = self.read_cl_array('linear_correction_fiducial')
            self.linear_correction = self.read_bin_windows('linear_correction_bin_window')
        else:
            self.linear_correction = None
        # if ini.hasKey('nuisance_params'):
        #     s = ini.relativeFileName('nuisance_params')
        #     self.nuisance_params = ParamNames(s)
        #     if ini.hasKey('calibration_param'):
        #         raise Exception('calibration_param not allowed with nuisance_params')
        #     if ini.hasKey('calibration_paramname'):
        #         self.calibration_param = ini.string('calibration_paramname')
        #     else:
        #         self.calibration_param = None
        # elif ini.string('calibration_param', ''):
        #     s = ini.relativeFileName('calibration_param')
        #     if '.paramnames' not in s:
        #         raise ValueError('calibration_param must be paramnames file unless '
        #                       'nuisance_params also specified')
        #     self.nuisance_params = ParamNames(s)
        #     self.calibration_param = self.nuisance_params.list()[0]
        # else:
        #     self.calibration_param = None
        # if ini.hasKey('log_calibration_prior'):
        #     self.log.warning('log_calibration_prior in .dataset ignored, '
        #                      'set separately in .yaml file')
        # self.aberration_coeff = ini.float('aberration_coeff', 0.0)

        self.map_cls = self.init_map_cls(self.nmaps_required, self.required_order)

    def get_loglike(self, class_input, likes_input, class_run):
        cls = class_run.lensed_cl()
        dls = {}
        ells = cls['ell']
        for k in cls.keys():
            if k in ['tt', 'te', 'ee']:
                dls[k] = cls[k] * ells * (ells+1) / 2. / np.pi * 1e12 * class_run.T_cmb()**2.
            elif k == 'pp':
                dls[k] = cls[k] * (ells * (ells+1))**2. / 2. / np.pi
            elif k in ['tp', 'ep']:
                dls[k] = cls[k] * (ells * (ells+1))**1.5 / 2. / np.pi * 1e6 * class_run.T_cmb()
        self.get_theory_map_cls(dls, likes_input)
        C = np.empty((self.nmaps, self.nmaps))
        big_x = np.empty(self.nbins_used * self.ncl_used)
        vecp = np.empty(self.ncl)
        chisq = 0
        if self.binned:
            binned_theory = self.get_binned_map_cls(self.map_cls)
        else:
            Cs = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
            for i in range(self.nmaps):
                for j in range(i + 1):
                    CL = self.map_cls[i, j]
                    if CL is not None:
                        Cs[:, i, j] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
                        Cs[:, j, i] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
        for b in range(self.nbins_used):
            if self.binned:
                self.elements_to_matrix(binned_theory[b, :], C)
            else:
                C[:, :] = Cs[b, :, :]
            if self.cl_noise is not None:
                C += self.noise_matrix[b]
            if self.like_approx == 'exact':
                chisq += self.exact_chi_sq(
                    C, self.bandpower_matrix[b], self.bin_min + b)
                continue
            elif self.like_approx == 'HL':
                try:
                    self.transform(
                        C, self.bandpower_matrix[b], self.fiducial_sqrt_matrix[b])
                except np.linalg.LinAlgError:
                    print("Likelihood computation failed.")
                    return -np.inf
            elif self.like_approx == 'gaussian':
                C -= self.bandpower_matrix[b]
            self.matrix_to_elements(C, vecp)
            big_x[b * self.ncl_used:(b + 1) * self.ncl_used] = vecp[
                self.cl_used_index]
        if self.like_approx == 'exact':
            return -0.5 * chisq
        return -0.5 * self.fast_chi_squared(self.covinv, big_x)

    def fast_chi_squared(self, covinv, x):
        return covinv.dot(x).dot(x)

    def typeIndex(self, field):
        return self.field_names.index(field)

    def PairStringToMapIndices(self, S):
        if len(S) == 2:
            if self.has_map_names:
                raise ValueError('CL names must use MAP1xMAP2 names')
            return self.map_names.index(S[0]), self.map_names.index(S[1])
        else:
            try:
                i = S.index(self.map_separator)
            except ValueError:
                raise ValueError('invalid spectrum name %s' % S)
            return self.map_names.index(S[0:i]), self.map_names.index(S[i + 1:])

    def PairStringToUsedMapIndices(self, used_index, S):
        i1, i2 = self.PairStringToMapIndices(S)
        i1 = used_index[i1]
        i2 = used_index[i2]
        if i2 > i1:
            return i2, i1
        else:
            return i1, i2

    def UseString_to_cols(self, L):
        cl_i_j = self.UseString_to_Cl_i_j(L, self.map_used_index)
        cols = -np.ones(cl_i_j.shape[1], dtype=int)
        for i in range(cl_i_j.shape[1]):
            i1, i2 = cl_i_j[:, i]
            if i1 == -1 or i2 == -1:
                continue
            ix = 0
            for ii in range(self.nmaps):
                for jj in range(ii + 1):
                    if ii == i1 and jj == i2:
                        cols[i] = ix
                    ix += 1
        return cols

    def UseString_to_Cl_i_j(self, S, used_index):
        if not isinstance(S, (list, tuple)):
            S = S.split()
        cl_i_j = np.zeros((2, len(S)), dtype=int)
        for i, p in enumerate(S):
            cl_i_j[:, i] = self.PairStringToUsedMapIndices(used_index, p)
        return cl_i_j

    def MapPair_to_Theory_i_j(self, order, pair):
        i = self.map_fields[order[pair[0]]]
        j = self.map_fields[order[pair[1]]]
        if i <= j:
            return i, j
        else:
            return j, i

    def Cl_used_i_j_name(self, pair):
        return self.Cl_i_j_name(self.used_map_order, pair)

    def Cl_i_j_name(self, names, pair):
        name1 = names[pair[0]]
        name2 = names[pair[1]]
        if self.has_map_names:
            return name1 + self.map_separator + name2
        else:
            return name1 + name2

    def get_cols_from_order(self, order):
        # Converts string Order = TT TE EE XY... or AAAxBBB AAAxCCC BBxCC
        # into indices into array of power spectra (and -1 if not present)
        cols = np.empty(self.ncl, dtype=int)
        cols[:] = -1
        names = order.strip().split()
        ix = 0
        for i in range(self.nmaps):
            for j in range(i + 1):
                name = self.Cl_used_i_j_name([i, j])
                if name not in names and i != j:
                    name = self.Cl_used_i_j_name([j, i])
                if name in names:
                    if cols[ix] != -1:
                        raise ValueError('get_cols_from_order: duplicate CL type')
                    cols[ix] = names.index(name)
                ix += 1
        return cols

    def elements_to_matrix(self, X, M):
        ix = 0
        for i in range(self.nmaps):
            M[i, 0:i] = X[ix:ix + i]
            M[0:i, i] = X[ix:ix + i]
            ix += i
            M[i, i] = X[ix]
            ix += 1

    def matrix_to_elements(self, M, X):
        ix = 0
        for i in range(self.nmaps):
            X[ix:ix + i + 1] = M[i, 0:i + 1]
            ix += i + 1

    def read_cl_array(self, file_stem, return_full=False):
        # read file of CL or bins (indexed by L)
        if file_stem == "cl_hat":
            filename = self.cl_hat_file
        elif file_stem == "linear_correction_fiducial":
            filename = self.linear_correction_fiducial_file
        cl = np.zeros((self.ncl, self.nbins_used))
        #order = ini.string(file_stem + '_order', '')
        order = ''
        if not order:
            incols = last_top_comment(filename)
            if not incols:
                raise ValueError('No column order given for ' + filename)
        else:
            incols = 'L ' + order
        cols = self.get_cols_from_order(incols)
        data = np.loadtxt(filename)
        Ls = data[:, 0].astype(int)
        if self.binned:
            Ls -= 1
        for i, L in enumerate(Ls):
            if self.bin_min <= L <= self.bin_max:
                for ix in range(self.ncl):
                    if cols[ix] != -1:
                        cl[ix, L - self.bin_min] = data[i, cols[ix]]
        if Ls[-1] < self.bin_max:
            raise ValueError('CMBLikes_ReadClArr: C_l file does not go up to maximum used: '
                          '%s', self.bin_max)
        if return_full:
            return incols.split(), data, cl
        else:
            return cl

    def read_bin_windows(self, file_stem):
        bins = BinWindows(self.pcl_lmin, self.pcl_lmax, self.nbins_used, self.ncl)
        if file_stem == "bin_window":
            in_cl = self.bin_window_in_order
            out_cl = self.bin_window_out_order
            windows = self.bin_window_files
        elif file_stem == "linear_correction_bin_window":
            in_cl = self.linear_correction_bin_window_in_order
            out_cl = self.linear_correction_bin_window_out_order
            windows = self.linear_correction_bin_window_files
        bins.cols_in = self.UseString_to_Cl_i_j(in_cl, self.map_required_index)
        bins.cols_out = self.UseString_to_cols(out_cl)
        norder = bins.cols_in.shape[1]
        if norder != bins.cols_out.shape[0]:
            raise ValueError('_in_order and _out_order must have same number of entries')
        bins.binning_matrix = np.zeros(
            (norder, self.nbins_used, self.pcl_lmax - self.pcl_lmin + 1))
        for b in range(self.nbins_used):
            window = np.loadtxt(windows % (b + 1 + self.bin_min))
            err = False
            for i, L in enumerate(window[:, 0].astype(int)):
                if self.pcl_lmin <= L <= self.pcl_lmax:
                    bins.binning_matrix[:, b, L - self.pcl_lmin] = window[i, 1:]
                else:
                    err = err or np.count_nonzero(window[i, 1:])
            if err:
                self.log.warning('%s %u outside pcl_lmin-cl_max range: %s' %
                                 (file_stem, b, windows % (b + 1)))
        #if ini.hasKey(file_stem + '_fix_cl_file'):
        #    raise LoggedError(self.log, 'fix_cl_file not implemented yet')
        return bins

    def init_map_cls(self, nmaps, order):
        if nmaps != len(order):
            raise ValueError('init_map_cls: size mismatch')

        class CrossPowerSpectrum:
            map_ij: List[int]
            theory_ij: List[int]
            CL: np.ndarray

        cls = np.empty((nmaps, nmaps), dtype=CrossPowerSpectrum)
        for i in range(nmaps):
            for j in range(i + 1):
                CL = CrossPowerSpectrum()
                cls[i, j] = CL
                CL.map_ij = [order[i], order[j]]
                CL.theory_ij = self.MapPair_to_Theory_i_j(order, [i, j])
                CL.CL = np.zeros(self.pcl_lmax - self.pcl_lmin + 1)
        return cls

    def ReadCovmat(self):
        """Read the covariance matrix, and the array of which CL are in the covariance,
        which then defines which set of bandpowers are used
        (subject to other restrictions)."""
        covmat_cl = self.covmat_cl
        self.full_cov = np.loadtxt(self.covmat_fiducial)
        covmat_scale = 1.0
        cl_in_index = self.UseString_to_cols(covmat_cl)
        self.ncl_used = np.sum(cl_in_index >= 0)
        self.cl_used_index = np.zeros(self.ncl_used, dtype=int)
        cov_cl_used = np.zeros(self.ncl_used, dtype=int)
        ix = 0
        for i, index in enumerate(cl_in_index):
            if index >= 0:
                self.cl_used_index[ix] = index
                cov_cl_used[ix] = i
                ix += 1
        if self.binned:
            num_in = len(cl_in_index)
            pcov = np.empty((self.nbins_used * self.ncl_used,
                             self.nbins_used * self.ncl_used))
            for binx in range(self.nbins_used):
                for biny in range(self.nbins_used):
                    pcov[binx * self.ncl_used: (binx + 1) * self.ncl_used,
                    biny * self.ncl_used: (biny + 1) * self.ncl_used] = (
                            covmat_scale * self.full_cov[
                        np.ix_((binx + self.bin_min) * num_in + cov_cl_used,
                               (biny + self.bin_min) * num_in + cov_cl_used)])
        else:
            raise ValueError('unbinned covariance not implemented yet')
        return pcov

    def diag_sigma(self):
        return np.sqrt(np.diag(self.full_cov))

    def get_binned_map_cls(self, Cls, corrections=True):
        band = self.bins.bin(Cls)
        if self.linear_correction is not None and corrections:
            band += self.linear_correction.bin(Cls) - self.fid_correction.T
        return band

    def get_theory_map_cls(self, Cls, data_params=None):
        for i in range(self.nmaps_required):
            for j in range(i + 1):
                CL = self.map_cls[i, j]
                combination = "".join([self.field_names[k] for k in CL.theory_ij]).lower()
                cls = Cls.get(combination)
                if cls is not None:
                    CL.CL[:] = cls[self.pcl_lmin:self.pcl_lmax + 1]
                else:
                    CL.CL[:] = 0
        self.adapt_theory_for_maps(self.map_cls, data_params or {})

    def adapt_theory_for_maps(self, cls, data_params):
        # if self.aberration_coeff:
        #     self.add_aberration(cls)
        # self.add_foregrounds(cls, data_params)
        if self.calibration_param is not None and self.calibration_param in data_params:
            for i in range(self.nmaps_required):
                for j in range(i + 1):
                    CL = cls[i, j]
                    if CL is not None:
                        if CL.theory_ij[0] <= 2 and CL.theory_ij[1] <= 2:
                            CL.CL /= data_params[self.calibration_param] ** 2
        else:
            pass

    @staticmethod
    def transform(C, Chat, Cfhalf):
        # HL transformation of the matrices
        if C.shape[0] == 1:
            rat = Chat[0, 0] / C[0, 0]
            C[0, 0] = (np.sign(rat - 1) *
                       np.sqrt(2 * np.maximum(0, rat - np.log(rat) - 1)) *
                       Cfhalf[0, 0] ** 2)
            return
        diag, U = np.linalg.eigh(C)
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        diag, rot = np.linalg.eigh(rot)
        diag = np.sign(diag - 1) * np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1))
        Cfhalf.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        rot.dot(U.T, C)


class BinWindows:
    cols_in: np.ndarray
    cols_out: np.ndarray
    binning_matrix: np.ndarray

    def __init__(self, lmin, lmax, nbins, ncl):
        self.lmin = lmin
        self.lmax = lmax
        self.nbins = nbins
        self.ncl = ncl

    def bin(self, theory_cl, cls=None):
        if cls is None:
            cls = np.zeros((self.nbins, self.ncl))
        for i, ((x, y), ix_out) in enumerate(zip(self.cols_in.T, self.cols_out)):
            cl = theory_cl[x, y]
            if cl is not None and ix_out >= 0:
                cls[:, ix_out] += np.dot(self.binning_matrix[i, :, :], cl.CL)
        return cls

# def last_top_comment(fname):
#     result = None
#     with open(fname, encoding="utf-8-sig") as f:
#         while x := f.readline():
#             if x := x.strip():
#                 if x[0] != '#':
#                     return result
#                 result = x[1:].strip()
#     return None

# New version of last_top_comment to avoid using walrus operator
# and to be compatible with Python 3.6+
def last_top_comment(fname):
    result = None
    with open(fname, encoding="utf-8-sig") as f:
        line = f.readline()
        while line:
            stripped = line.strip()
            if stripped:
                if stripped[0] != '#':
                    return result
                result = stripped[1:].strip()
            line = f.readline()
    return None
