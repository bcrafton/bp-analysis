
import numpy as np
import matplotlib.pyplot as plt

bp = np.load('vgg64_bp.npy').item()
lel = np.load('vgg64_lel.npy').item()

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8']
# layers = ['conv1', 'conv2']

for l in layers:
    lel_l = lel[l] 
    bp_l = bp[l]
    scale = np.std(bp[l]) / np.std(lel[l])
    lel_l = scale * lel_l
    
    bp_std = np.std(bp_l, axis=(0, 1, 3))
    lel_std = np.std(lel_l, axis=(0, 1, 3))
    
    bp_mean = np.mean(bp_l, axis=(0, 1, 2))
    lel_mean = np.mean(lel_l, axis=(0, 1, 2))
    
    plt.hist(bp_std, bins=100, label='bp')
    plt.hist(lel_std, bins=100, label='lel')
    plt.legend()
    plt.show()
