'''
Author: Benedict Diederich
## This is the convolutional Neural network which tries to find an implicit model
# between complex object transmission functions and its optimized illumination 
# source shapes which enhance the phase-contrast in the image
# 
# The software is for private use only and gives no guarrantee, that it's
# working as it should! 
# 
#
# Written by Benedict Diederich, benedict.diederich@leibniz-ipht.de
# www.nanoimaging.de
# License: GPL v3 or later.
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import h5py 
from matplotlib import pyplot as plt
from scipy import io
import scipy as scipy
from scipy import stats
import modeldef_conv_2D as mod
import input_data_v1 as indata
import time
import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat



# logpath for tensorboard
logs_path = "./logs/nn_logs/"
filename_pb = 'expert-graph_CN.pb'
  
inputmat_name = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Python/Beamerscope_TENSORFLOW/Beamerscope_IllOpt/nninputs.mat'

outputmat_name = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Python/Beamerscope_TENSORFLOW/Beamerscope_IllOpt/nnoutputs.mat'


# session.close()
tf.reset_default_graph()


#################recreate model and safe for android###########################
    

model_filename = logs_path + filename_pb 
    
# taken from https://github.com/tensorflow/tensorflow/issues/616
sess = tf.InteractiveSession()
print("load graph")
with gfile.FastGFile(model_filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    print("map variables")
  
    weights = {
            # [5, 5, 1, 32] = kernel_m, kernel_n, feature_n_old, feature_n_new           
            # 5x5 conv, 1 input, 32 outputs
            'wc1': sess.graph.get_tensor_by_name("wc1:0"),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': sess.graph.get_tensor_by_name("wc2:0"),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': sess.graph.get_tensor_by_name("wd1:0"),    
            # 1024 inputs, 10 outputs (class prediction)
            'wout': sess.graph.get_tensor_by_name("wout:0")
            }
    
    biases = {
             'bc1': sess.graph.get_tensor_by_name("bc1:0"),
             'bc2': sess.graph.get_tensor_by_name("bc2:0"),
             'bd1': sess.graph.get_tensor_by_name("bd1:0"),
             'bout': sess.graph.get_tensor_by_name("bout:0")
    }
    

    # extract/backup all learned variables
    wc1 = weights['wc1'].eval()
    wc2 = weights['wc2'].eval()
    wd1 = weights['wd1'].eval()
    out1 = weights['wout'].eval()
    
    bc1 = biases['bc1'].eval()
    bc2 = biases['bc2'].eval()
    bd1 = biases['bd1'].eval()
    bout = biases['bout'].eval()
    

sess.close()
# Create new graph for exporting
g_2 = tf.Graph()

with g_2.as_default():
    # Store layers weight & bias
    weights_store = {
            # [5, 5, 1, 32] = kernel_m, kernel_n, feature_n_old, feature_n_new           
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.constant(wc1, name = 'wc1'),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.constant(wc2, name = 'wc2'),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.constant(wd1, name = 'wd1'),
            # 1024 inputs, 10 outputs (class prediction)
            'wout': tf.constant(out1, name = 'wout')
            }
        
        
    biases_store = {
        'bc1': tf.constant(bc1, name = 'bc1'),
        'bc2': tf.constant(bc2, name = 'bc2'),
        'bd1': tf.constant(bd1, name = 'bd1'),
        'bout': tf.constant(bout, name = 'bout')
    }

    # tf Graph input
    n_length = 128
    x_store = tf.placeholder(tf.float32, [None, n_length, n_length, 2], name = 'input')
    keep_prob_store = tf.constant(1., tf.float32, name = 'dropout-factor') #dropout (keep probability)
    is_training = tf.constant(False)
    
    # Construct model
    y_store = mod.conv_net(x_store, weights_store, biases_store, keep_prob_store, is_training, generation_phase = True)
    
    sess_2 = tf.InteractiveSession()
    init_2 = tf.global_variables_initializer();
    sess_2.run(init_2)
    
    # test learned network with random spectrum
    testobj = np.concatenate(((np.random.rand(1, n_length, n_length, 1))*0.001, 2*(np.random.rand(1, n_length, n_length, 1))-0), 3)
    y_pred = sess_2.run(y_store, feed_dict={x_store: testobj})
    plt.plot(y_pred.T)
    plt.show()    
    
    


