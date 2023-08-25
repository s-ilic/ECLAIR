import numpy as np

### BAO "2014" data (used in Planck 2015 as ext. data)
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
    # BOSS LOWZ & CMASS DR10&11 from 1312.4877
    z, data, error = 0.32, 8.47, 0.17
    da = class_run.angular_distance(z)
    dr = z / class_run.Hubble(z)
    dv = (da**2. * (1 + z)**2. * dr)**(1. / 3.)
    theo = dv / rs
    lnl += -0.5 * (theo - data)**2. / error**2.
    z, data, error = 0.57, 13.77, 0.13
    da = class_run.angular_distance(z)
    dr = z / class_run.Hubble(z)
    dv = (da**2. * (1 + z)**2. * dr)**(1. / 3.)
    theo = dv / rs
    lnl += -0.5 * (theo - data)**2. / error**2.
    # SDSS DR7 MGS from 1409.3242
    z, data, error = 0.15, 4.47, 0.16
    da = class_run.angular_distance(z)
    dr = z / class_run.Hubble(z)
    dv = (da**2. * (1 + z)**2. * dr)**(1. / 3.)
    theo = dv / rs
    lnl += -0.5 * (theo - data)**2. / error**2.
    # Return log(like)
    return lnl
