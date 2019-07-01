
import numpy as np
import tensorflow as tf

from conv_utils import conv_output_length
from conv_utils import conv_input_length
from conv_utils import get_pad

'''
def combine_filter(f1, f2, stride=1):

    h1, w1, fin1, fout1 = np.shape(f1)
    h2, w2, fin2, fout2 = np.shape(f2)
    
    oh = h1 + 2*(h2 // 2) * stride
    ow = w1 + 2*(w2 // 2) * stride
    ofin = fin1
    ofout = fout2
    fout = np.zeros(shape=(oh, ow, ofin, ofout))
    
    for x in range(h2):
        for y in range(w2):
            for c1 in range(fin2):
                for c2 in range(fout2):
                    sh = x * stride ; eh = x * stride + h1
                    sw = y * stride ; ew = y * stride + w1
                    fout[sh:eh, sw:ew, :, c2] = fout[sh:eh, sw:ew, :, c2] + f2[x][y][c1][c2] * f1[:, :, :, c1]

    return fout
'''
    
def combine_filter(f1, f2, stride=1):

    h1, w1, fin1, fout1 = np.shape(f1)
    h2, w2, fin2, fout2 = np.shape(f2)
    
    oh = h1 + 2*(h2 // 2) * stride
    ow = w1 + 2*(w2 // 2) * stride
    
    ofin = fin1
    ofout = fout2
    
    fout = np.zeros(shape=(oh, ow, ofin, ofout))
    
    for x in range(h2):
        for y in range(w2):
            for c in range(fout2):
                sh = x * stride ; eh = x * stride + h1
                sw = y * stride ; ew = y * stride + w1
                fout[sh:eh, sw:ew, :, c] += np.sum(f2[x, y, :, c] * f1, axis=3)

    return fout

def combine_filter_dw(f1, f2, stride=1):

    h1, w1, fin1, fout1 = np.shape(f1)
    h2, w2, fin2, fout2 = np.shape(f2)
    assert(fout2 == 1)
    
    oh = h1 + 2*(h2 // 2) * stride
    ow = w1 + 2*(w2 // 2) * stride
    ofin = fin1
    ofout = fin2
    fout = np.zeros(shape=(oh, ow, ofin, ofout))
    
    # dic = {}
    
    for x in range(h2):
        for y in range(w2):
            for c in range(fin2):
                sh = x * stride ; eh = x * stride + h1
                sw = y * stride ; ew = y * stride + w1
                
                # fout[sh:eh, sw:ew, :, c2] += f2[x][y][c1][c2] * f1[:, :, :, c1]
                
                fout[sh:eh, sw:ew, :, c] += f2[x][y][c][0] * f1[:, :, :, c]
                # key = (sh, eh, sw, ew, c)                
                # print (key in dic.keys())
                # dic[key] = key
                
                # print (x, y, c)
                # print (np.shape(fout[sh:eh, sw:ew, :, c]), np.shape(f1[:, :, :, c]), np.shape(f2[x][y][c][0]))

    return fout

'''
def combine_filter(filters1, filters2, stride=1):

    filter_size, _, _, _ = np.shape(filters2)
    pad = get_pad('full', filter_size)
    filters1_pad = np.pad(filters1, [[pad, pad], [pad, pad], [0, 0], [0, 0]], mode='constant')
    
    f1_shape = list(np.shape(filters1_pad))
    f2_shape = list(np.shape(filters2))
    
    filters1_ph = tf.placeholder(tf.float32, f1_shape)
    filters2_ph = tf.placeholder(tf.float32, f2_shape)
    
    image = tf.transpose(filters1_ph, (2, 0, 1, 3))
    filters = filters2_ph
    conv = tf.nn.conv2d(image, filters, [1,stride,stride,1], 'VALID')
    combined = tf.transpose(conv, (1, 2, 0, 3))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    [ret] = sess.run([combined], feed_dict={filters1_ph: filters1_pad, filters2_ph: filters2})
    
    return ret
'''

'''
def combine_filter(filters1, filters2, stride=1):

    h1, w1, fin1, fout1 = np.shape(filters1)
    h2, w2, fin2, fout2 = np.shape(filters2)
    
    batch_size = fout2
    input_h = h1 + 2*(h2 // 2) * stride
    input_w = w1 + 2*(w2 // 2) * stride
    input_fin = fin1
    
    input_sizes = [batch_size, input_h, input_w, input_fin]
    
    ###########

    filters1_shape = list(np.shape(filters1))
    filters2_shape = list(np.shape(filters2))

    filters1_ph = tf.placeholder(tf.float32, filters1_shape)
    filters2_ph = tf.placeholder(tf.float32, filters2_shape)
    
    backwards = tf.transpose(filters2_ph, (3, 0, 1, 2))
    filters = filters1_ph
    dconv = tf.nn.conv2d_backprop_input(input_sizes=input_sizes, filter=filters, out_backprop=backwards, strides=[1,stride,stride,1], padding='VALID')
    combined = tf.transpose(dconv, (1, 2, 3, 0))
    
    ###########

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    [ret] = sess.run([combined], feed_dict={filters1_ph: filters1, filters2_ph: filters2})
    
    return ret
'''

'''
def combine_filter(filters1, filters2, stride=1):

    h1, w1, fin1, fout1 = np.shape(filters1)
    h2, w2, fin2, fout2 = np.shape(filters2)
    
    batch_size = fout2
    input_h = h1 + 2*(h2 // 2) * stride
    input_w = w1 + 2*(w2 // 2) * stride
    input_fin = fin1
    
    input_sizes = [batch_size, input_h, input_w, input_fin]
    
    ###########

    filters1_shape = list(np.shape(filters1))
    filters2_shape = list(np.shape(filters2))

    filters1_ph = tf.placeholder(tf.float32, filters1_shape)
    filters2_ph = tf.placeholder(tf.float32, filters2_shape)
    
    backwards = tf.transpose(filters2_ph, (3, 0, 1, 2))
    filters = filters1_ph
    dconv = tf.nn.conv2d_backprop_input(input_sizes=input_sizes, filter=filters, out_backprop=backwards, strides=[1,stride,stride,1], padding='VALID')
    combined = tf.transpose(dconv, (1, 2, 3, 0))
    
    ###########

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    [ret] = sess.run([combined], feed_dict={filters1_ph: filters1, filters2_ph: filters2})
    
    return ret
'''

'''
def combine_filter_dw(filters1, filters2, stride=1):

    h1, w1, fin1, fout1 = np.shape(filters1)
    h2, w2, fin2, fout2 = np.shape(filters2)
    
    batch_size = fout2
    input_h = h1 + 2*(h2 // 2) * stride
    input_w = w1 + 2*(w2 // 2) * stride
    input_fin = fin1
    
    input_sizes = [batch_size, input_h, input_w, input_fin]
    
    ###########

    filters1_shape = list(np.shape(filters1))
    filters2_shape = list(np.shape(filters2))

    filters1_ph = tf.placeholder(tf.float32, filters1_shape)
    filters2_ph = tf.placeholder(tf.float32, filters2_shape)
    
    backwards = tf.transpose(filters2_ph, (3, 0, 1, 2))
    filters = filters1_ph
    dconv = tf.nn.conv2d_backprop_input(input_sizes=input_sizes, filter=filters, out_backprop=backwards, strides=[1,stride,stride,1], padding='VALID')
    combined = tf.transpose(dconv, (1, 2, 3, 0))
    
    ###########

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    [ret] = sess.run([combined], feed_dict={filters1_ph: filters1, filters2_ph: filters2})
    
    return ret
'''




    
