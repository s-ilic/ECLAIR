import numpy as np

### From Riess et al., 1903.07603
def get_loglike(class_input, likes_input, class_run):
    return -0.5 * (class_run.h() * 100 - 74.03)**2. / 1.42**2.
