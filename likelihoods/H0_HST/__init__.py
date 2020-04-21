import numpy as np

### From Riess et al., 1103.2976
def get_loglike(class_input, likes_input, class_run):
    return -0.5 * (class_run.h() * 100 - 73.8)**2. / 2.4**2.
