import numpy as np
import torch

def random_walk(length, interval=1, p=0.5, amplitude=1, acceleration=0.8):
    xvals = np.arange(0, length) # *interval
    split = np.random.choice(np.arange(int(0.3*length), int(0.7*length)))
    pre_intervention = np.random.choice((-1,1), split, p=(1-p, p))
    if np.random.choice((-1,1)) > 0:
        accel = (1-acceleration, acceleration)
    else:
        accel = (acceleration, 1-acceleration)
    post_intervention = np.random.choice((-1,1), (length-split), p=accel)
    
    yvals = np.cumsum(np.concatenate([pre_intervention, post_intervention]))
    
    return interval*xvals, yvals