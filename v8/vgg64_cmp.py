
import numpy as np
np.set_printoptions(precision=2)
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
ref = np.load('ref.npy')

print (np.max(act - ref), np.max(act), np.max(ref))

#####











