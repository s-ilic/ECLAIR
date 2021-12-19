import itertools
import os, sys
import re
from typing import Optional, Sequence
import numpy as np

default_spectra_list = [
    "90_Ex90_E",
    "90_Tx90_E",
    "90_Ex150_E",
    "90_Tx150_E",
    "90_Ex220_E",
    "90_Tx220_E",
    "150_Ex150_E",
    "150_Tx150_E",
    "150_Ex220_E",
    "150_Tx220_E",
    "220_Ex220_E",
    "220_Tx220_E",
]

self_bin_min = 1
self_bin_max = 44
self_windows_lmin = 1
self_windows_lmax = 3200
self_aberration_coefficient = -0.0004826
self_super_sample_lensing = True
self_poisson_switch = True
self_dust_switch = True
self_beam_cov_scaling = 1.0
self_spectra_to_fit = default_spectra_list
# SPT-3G Y1 EE/TE Effective band centres for polarised galactic dust.
self_nu_eff_list = {90: 9.670270e01, 150: 1.499942e02, 220: 2.220433e02}

self_data_folder = "data"
# Bandpower file
self_bp_file = "SPT3G_Y1_EETE_bandpowers.dat"
# Covariance file
self_cov_file = "SPT3G_Y1_EETE_covariance.dat"
# Beam covariance file
self_beam_cov_file = "SPT3G_Y1_EETE_beam_covariance.dat"
# Calibration (mapT, mapP) covariance
self_calib_cov_file = "SPT3G_Y1_EETE_cal_covariance.dat"
# Windows directory
self_window_dir = "windows"

data_file_path = os.path.dirname(os.path.realpath(sys.argv[0]))
data_file_path += '/likelihoods/SPT3G_2020'

self_data_folder = os.path.join(data_file_path, self_data_folder)

self_nbins = self_bin_max - self_bin_min + 1
if self_nbins < 1:
    raise ValueError(f"Selected an invalid number of bandpowers ({self_nbins})")

if self_windows_lmin < 1 or self_windows_lmin >= self_windows_lmax:
    raise ValueError("Invalid ell ranges for SPTPol")

# Read in bandpowers (remove index column)
self_bandpowers = np.loadtxt(os.path.join(self_data_folder, self_bp_file), unpack=True)[1:]

# Read in covariance
self_cov = np.loadtxt(os.path.join(self_data_folder, self_cov_file))

# Read in beam covariance
self_beam_cov = np.loadtxt(os.path.join(self_data_folder, self_beam_cov_file))

# Read in windows
self_windows = np.array(
    [
        np.loadtxt(
            os.path.join(self_data_folder, self_window_dir, f"window_{i}.txt"), unpack=True
        )[1:]
        for i in range(self_bin_min, self_bin_max + 1)
    ]
)

# Compute spectra/cov indices given spectra to fit
vec_indices = np.array([default_spectra_list.index(spec) for spec in self_spectra_to_fit])
self_bandpowers = self_bandpowers[vec_indices].flatten()
self_windows = self_windows[:, vec_indices, :]
cov_indices = np.array(
    [np.arange(i * self_nbins, (i + 1) * self_nbins) for i in vec_indices]
)
cov_indices = cov_indices.flatten()
# Select spectra/cov elements given indices
self_cov = self_cov[np.ix_(cov_indices, cov_indices)]
self_beam_cov = self_beam_cov[np.ix_(cov_indices, cov_indices)] * self_beam_cov_scaling

# Compute cross-spectra frequencies and mode given the spectra name to fit
r = re.compile("(.+?)_(.)x(.+?)_(.)")
self_cross_frequencies = [r.search(spec).group(1, 3) for spec in self_spectra_to_fit]
self_cross_spectra = ["".join(r.search(spec).group(2, 4)) for spec in self_spectra_to_fit]
self_frequencies = sorted(
    {float(freq) for freqs in self_cross_frequencies for freq in freqs}
)

# Read in calibration covariance and select mode/frequencies
# The order of the cal covariance is T90, T150, T220, E90, E150, E220
calib_cov = np.loadtxt(os.path.join(self_data_folder, self_calib_cov_file))
cal_indices = np.array([[90.0, 150.0, 220.0].index(freq) for freq in self_frequencies])
if "TE" not in self_cross_spectra:
    # Only polar calibrations shift by 3
    cal_indices += 3
else:
    cal_indices = np.concatenate([cal_indices, cal_indices + 3])
calib_cov = calib_cov[np.ix_(cal_indices, cal_indices)]
self_inv_calib_cov = np.linalg.inv(calib_cov)
self_calib_params = np.array(
    ["map{}cal{}".format(*p) for p in itertools.product(["T", "P"], [90, 150, 220])]
)[cal_indices]
self_log.debug(f"Calibration parameters: {self_calib_params}")

self_lmin = self_windows_lmin
self_lmax = self_windows_lmax + 1  # to match fortran convention
self_ells = np.arange(self_lmin, self_lmax)

# def loglike(self, dlte, dlee, **params_values):
def get_loglike(class_input, likes_input, class_run):

    T_CMB = 2.72548  # CMB temperature
    h = 6.62606957e-34  # Planck's constant
    kB = 1.3806488e-23  # Boltzmann constant
    Ghz_Kelvin = h / kB * 1e9

    # Recover Cl_s from CLASS
    fl = class_run.lensed_cl()['ell'] * (class_run.lensed_cl()['ell'] + 1.) / (2. * np.pi)
    dlte = fl * class_run.lensed_cl()['te'] * 1e12 * T_CMB**2.
    dlee = fl * class_run.lensed_cl()['ee'] * 1e12 * T_CMB**2.

    # Planck function normalised to 1 at nu0
    def Bnu(nu, nu0, T):
        return (
            (nu / nu0) ** 3
            * (np.exp(Ghz_Kelvin * nu0 / T) - 1)
            / (np.exp(Ghz_Kelvin * nu / T) - 1)
        )

    # Derivative of Planck function normalised to 1 at nu0
    def dBdT(nu, nu0, T):
        x0 = Ghz_Kelvin * nu0 / T
        x = Ghz_Kelvin * nu / T

        dBdT0 = x0 ** 4 * np.exp(x0) / (np.exp(x0) - 1) ** 2
        dBdT = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2

        return dBdT / dBdT0

    lmin, lmax = self_lmin, self_lmax
    ells = np.arange(lmin, lmax + 2)

    dbs = np.empty_like(self_bandpowers)
    for i, (cross_spectrum, cross_frequency) in enumerate(
        zip(self_cross_spectra, self_cross_frequencies)
    ):
        dl_cmb = dlee if cross_spectrum == "EE" else dlte

        # Calculate derivatives for this position in parameter space.
        cl_derivative = dl_cmb[ells] * 2 * np.pi / (ells * (ells + 1))
        cl_derivative = 0.5 * (cl_derivative[2:] - cl_derivative[:-2])

        # Add CMB
        dls = dl_cmb[self_ells]

        # Add super sample lensing
        # (In Cl space) SSL = -k/l^2 d/dln(l) (l^2Cl) = -k(l*dCl/dl + 2Cl)
        if self_super_sample_lensing:
            kappa = likes_input["kappa"]
            dls += -kappa * (
                self_ells ** 2 * (self_ells + 1) / (2 * np.pi) * cl_derivative
                + 2 * dl_cmb[self_ells]
            )

        # Aberration correction
        # AC = beta*l(l+1)dCl/dln(l)/(2pi)
        # Note that the CosmoMC internal aberration correction and the SPTpol Henning likelihood differ
        # CosmoMC uses dCl/dl, Henning et al dDl/dl
        # In fact, CosmoMC is correct:
        # https://journals-aps-org.eu1.proxy.openathens.net/prd/pdf/10.1103/PhysRevD.89.023003
        dls += (
            -self_aberration_coefficient
            * cl_derivative
            * self_ells ** 2
            * (self_ells + 1)
            / (2 * np.pi)
        )

        # Simple poisson foregrounds
        # This is any poisson power. Meant to describe both radio galaxies and DSFG. By giving each frequency combination an amplitude
        # to play with this gives complete freedom to the data
        if self_poisson_switch and cross_spectrum == "EE":
            Dl_poisson = likes_input["Dl_Poisson_{}x{}".format(*cross_frequency)]
            dls += self_ells * (self_ells + 1) * Dl_poisson / (3000 * 3001)

        # Polarised galactic dust
        if self_dust_switch:
            TDust = likes_input["TDust"]
            ADust = likes_input[f"ADust_{cross_spectrum}_150"]
            AlphaDust = likes_input[f"AlphaDust_{cross_spectrum}"]
            BetaDust = likes_input[f"BetaDust_{cross_spectrum}"]
            dfs = (
                lambda beta, temp, nu0, nu: (nu / nu0) ** beta
                * Bnu(nu, nu0, temp)
                / dBdT(nu, nu0, T_CMB)
            )
            dust = ADust * (self_ells / 80) ** (AlphaDust + 2)
            for freq in cross_frequency:
                dust *= dfs(BetaDust, TDust, 150, self_nu_eff_list.get(int(freq)))
            dls += dust

        # Scale by calibration
        if cross_spectrum == "EE":
            # Calibration for EE: 1/(Ecal_1*Ecal_2) since we matched the EE spectrum to Planck's
            calibration = 1.0
            for freq in cross_frequency:
                calibration /= likes_input[f"mapPcal{freq}"]
        if cross_spectrum == "TE":
            # Calibration for TE: 0.5*(1/(Tcal_1*Ecal_2) + 1/(Tcal_2*Ecal_1))
            freq1, freq2 = cross_frequency
            calibration = 0.5 * (
                1
                / (likes_input[f"mapTcal{freq1}"] * likes_input[f"mapPcal{freq2}"])
                + 1
                / (likes_input[f"mapTcal{freq2}"] * likes_input[f"mapPcal{freq1}"])
            )
        dls *= calibration

        # Binning via window and concatenate
        dbs[i * self_nbins : (i + 1) * self_nbins] = self_windows[:, i, :] @ dls

    # Take the difference to the measured bandpower
    delta_cb = dbs - self_bandpowers

    # Construct the full covariance matrix
    cov_w_beam = self_cov + self_beam_cov * np.outer(dbs, dbs)

    chi2 = delta_cb @ np.linalg.inv(cov_w_beam) @ delta_cb
    sign, slogdet = np.linalg.slogdet(cov_w_beam)

    # Add calibration prior
    delta_cal = np.log(np.array([likes_input[p] for p in self_calib_params]))
    cal_prior = delta_cal @ self_inv_calib_cov @ delta_cal

    return -0.5 * (chi2 + slogdet + cal_prior)
