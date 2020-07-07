import numpy as np


# Redshift and measurements for for SDSS DR12 Consensus BAO data (1607.03155)
DR12_z = np.array([0.38, 0.38, 0.51, 0.51, 0.61, 0.61])
DR12_data = np.array([1512.39, 81.2087, 1975.22, 90.9029, 2306.68, 98.9647])
DR12_rsdrag_fid = 147.78

# Covariance matrix for SDSS DR12 Consensus BAO data (1607.03155)
DR12_cov_mat = np.array(
    [
        [624.707, 23.729, 325.332, 8.34963, 157.386, 3.57778],
        [23.729, 5.60873, 11.6429, 2.33996, 6.39263, 0.968056],
        [325.332, 11.6429, 905.777, 29.3392, 515.271, 14.1013],
        [8.34963, 2.33996, 29.3392, 5.42327, 16.1422, 2.85334],
        [157.386, 6.39263, 515.271, 16.1422, 1375.12, 40.4327],
        [3.57778, 0.968056, 14.1013, 2.85334, 40.4327, 6.25936],
    ]
)
DR12_icov_mat = np.linalg.inv(DR12_cov_mat)


### BAO "2018" data (used in Planck 2018 as ext. data)
def get_loglike(class_input, likes_input, class_run):
    lnl = 0.
    rs = class_run.rs_drag()
    # 6DF from 1106.3366
    z, data, error = 0.106, 0.327, 0.015
    da = class_run.angular_distance(z)
    dr = z / class_run.Hubble(z)
    dv = (da**2. * (1 + z)**2. * dr)**(1. / 3.)
    theo = rs / dv
    lnl += -0.5 * (theo - data)**2. / error**2.
    # SDSS DR7 MGS from 1409.3242
    z, data, error = 0.15, 4.47, 0.16
    da = class_run.angular_distance(z)
    dr = z / class_run.Hubble(z)
    dv = (da**2. * (1 + z)**2. * dr)**(1. / 3.)
    theo = dv / rs
    lnl += -0.5 * (theo - data)**2. / error**2.
    # SDSS DR12 Consensus BAO from 1607.03155
    theo = np.zeros(6)
    for i in range(3):
        DMz = class_run.angular_distance(DR12_z[i*2]) * (1. + DR12_z[i*2])
        Hz = class_run.Hubble(DR12_z[i*2]) * 299792.458
        theo[i*2] = DMz / rs * DR12_rsdrag_fid
        theo[i*2+1] = Hz * rs / DR12_rsdrag_fid
    diff = DR12_data - theo
    lnl += -0.5 * np.dot(np.dot(diff, DR12_icov_mat), diff)
    # Return log(like)
    return lnl
