import numpy as np

### From Riess et al., 1801.01120
def get_loglike(class_input, likes_input, class_run):
    return -0.5 * (class_run.h() * 100 - 73.48)**2. / 1.66**2.
