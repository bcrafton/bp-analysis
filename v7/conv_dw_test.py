
import numpy as np

x = np.load('imagenet64_example.npy')
x = np.reshape(x[0], (1, 64, 64, 3))
x = x / 255.
x = x[0, 0:4, 0:4, :]
x = np.pad(x, [[1,0], [1,0], [0,0]], mode='constant')
assert(np.shape(x) == (5,5,3))

#####

weights = np.load('mobile64_bp.npy').item()
conv1_filters = weights['block1_conv']
conv2_dw_filters = weights['block2_conv_block_dw_conv_dw']

conv1_filters = np.absolute(conv1_filters)
conv2_dw_filters = np.absolute(conv2_dw_filters)

#####

out = np.zeros(shape=(3, 3))
out[0][0] = np.sum(conv1_filters[:, :, :, 0] * x[0:3, 0:3, :])
out[0][1] = np.sum(conv1_filters[:, :, :, 0] * x[0:3, 1:4, :])
#out[0][2] = np.sum(conv1_filters[:, :, :, 0] * x[0:3, 2:5, :])

out[1][0] = np.sum(conv1_filters[:, :, :, 0] * x[1:4, 0:3, :])
out[1][1] = np.sum(conv1_filters[:, :, :, 0] * x[1:4, 1:4, :])
#out[1][2] = np.sum(conv1_filters[:, :, :, 0] * x[1:4, 2:5, :])

#out[2][0] = np.sum(conv1_filters[:, :, :, 0] * x[2:5, 0:3, :])
#out[2][1] = np.sum(conv1_filters[:, :, :, 0] * x[2:5, 1:4, :])
#out[2][2] = np.sum(conv1_filters[:, :, :, 0] * x[2:5, 2:5, :])
#print (out)

#####

y = out
y = np.pad(y, [[1,0], [1,0]], mode='constant')

out = np.sum(conv2_dw_filters[:, :, 0, 0] * y[0:3, 0:3])
print (out)

#####

merged_filters = np.zeros(shape=(5, 5, 3))

#merged_filters[0:3, 0:3, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[0, 0, 0, 0]
#merged_filters[0:3, 1:4, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[0, 1, 0, 0]
#merged_filters[0:3, 2:5, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[0, 2, 0, 0]

#merged_filters[1:4, 0:3, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[0, 0, 0, 0]
merged_filters[1:4, 1:4, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[1, 1, 0, 0]
merged_filters[1:4, 2:5, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[1, 2, 0, 0]

#merged_filters[2:5, 0:3, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[0, 0, 0, 0]
merged_filters[2:5, 1:4, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[2, 1, 0, 0]#
merged_filters[2:5, 2:5, :] += conv1_filters[:, :, :, 0] * conv2_dw_filters[2, 2, 0, 0]

#####

x = np.load('imagenet64_example.npy')
x = np.reshape(x[0], (1, 64, 64, 3))
x = x / 255.
x = x[0, 0:3, 0:3, :]
x = np.pad(x, [[2,0], [2,0], [0,0]], mode='constant')
assert(np.shape(x) == (5,5,3))

#####

out = np.sum(merged_filters * x)
print (out)









