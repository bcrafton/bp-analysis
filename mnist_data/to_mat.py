
import numpy as np
import scipy.io as sio

data = {}
data['ones'] = np.load('ones.npy')
data['twos'] = np.load('twos.npy')
data['conv_ones'] = np.load('conv_ones.npy')
data['conv_twos'] = np.load('conv_twos.npy')
sio.savemat('data.mat', data)
