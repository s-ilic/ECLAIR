import numpy as np

### From Freedman et al., 2002.01550 
def get_loglike(class_input, likes_input, class_run):
    return -0.5 * (class_run.h() * 100 - 69.6)**2. / 2.5**2.
