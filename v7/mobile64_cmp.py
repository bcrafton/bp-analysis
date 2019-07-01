
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from conv import conv

from combine_filter import combine_filter
from combine_filter import combine_filter_dw

from viz_filter import viz_filter
from viz_filter import viz_filter_3_channels
from viz_filter import viz_filter_1_channels

#####

act = np.load('act.npy')
_, h, w, _ = np.shape(act)

ref = np.load('ref.npy')
ref = ref[:, 0:h, 0:w, :]

print (np.max(act - ref), np.max(act), np.max(ref))

#####











