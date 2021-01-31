import scipy.io
import numpy as np
from consensus_thresh import fcn_group_bins


dat = scipy.io.loadmat('/path-to-data/dataDistance.mat')

hemiid = np.array(dat['hemiid'] == 2)
G, Gc = fcn_group_bins(np.array(dat['A']),np.array(dat['dist']),
                       hemiid,41)