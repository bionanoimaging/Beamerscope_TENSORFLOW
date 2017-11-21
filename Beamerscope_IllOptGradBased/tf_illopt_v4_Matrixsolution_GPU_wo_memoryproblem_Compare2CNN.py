'''
Author: Benedict Diederich
%% Generate a dataset with complex objects and its corresponding optimized
% illumination shapes using the TCC
% 
% The software is for private use only and gives no guarrantee, that it's
% working as it should! 
% 
%
% Written by Benedict Diederich, benedict.diederich@leibniz-ipht.de
% www.nanoimaging.de
% License: GPL v3 or later.

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
from scipy import stats
import time
import scipy as scipy
from scipy import ndimage

import tensorflow as tf
import numpy as np
import h5py 
from matplotlib import pyplot as plt
from scipy import io
import scipy as scipy
from scipy import stats
import modeldef_conv_2D as mod
import time
import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat


#def contrastperpixel( grayImage ):
## contrastperpixel Contrast per Pixel Calculation from an input image
##   Detailed explanation goes here
#kernel = np.array((-1, -1, -1, -1, 8, -1, -1, -1))/8
#diffImage = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(grayImage, 0),3), kernel, strides=[1, 1, 1], padding='SAME')
#cpp = mean2(diffImage);
#
#end



def getSegment(xopt):
    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho

    kx = np.linspace(-1.5, 1.5, 100)
    ky = kx
    XX, YY  = np.meshgrid(kx, ky)
    
    mr=np.sqrt(XX**2+YY**2)
    mtheta, mr =  cart2pol(XX, YY)
    Po=1.*(mr<=1)
    n_segment = 12;     # number of colour segments
   
    Ic = np.zeros(Po.shape)
    
    for i in range(0, xopt.shape[0]):
        Isegment = np.zeros(Po.shape)
                
        # i-th segment in one of the annuli
        i_segment = np.mod(i, n_segment)
    
        if (np.int16(i/n_segment) == 0):
            NAc_i = 0;
            NAc_o = 0.25;
        elif (np.int16(i/n_segment) == 1):
            NAc_i = 0.25;
            NAc_o = 0.5;
        elif (np.int16(i/n_segment) == 2):
            NAc_i = 0.5;
            NAc_o = .75;
        elif (np.int16(i/n_segment) == 3):
            NAc_i = 0.75;
            NAc_o = 1;
        
          
        # Creating the annullar shape 0,1,2,3
        Isegment= (1.*(mr>=NAc_i) * 1.*(mr<=NAc_o)) #Inner and Outer radius.
            
    
        # scale rotational symmetric ramp 0..1
        segment_area = (mtheta)/np.max(mtheta) * np.round(n_segment/2) + np.round(n_segment/2);
        
        # each segment results from the threshold of the grayvalues
        # filtered by the annular shape of the illumination sector
        # 0,1,2
        
        # this is due to aliasing of the pixelated source, otherwise
        # there will be a gap in the source shape
        if(i_segment == n_segment-1):
            segment_area = 1.*(segment_area >= i_segment) * 1.*(segment_area < (i_segment+1)*1.00001)
        else:
            segment_area = 1.*(segment_area >= i_segment) * 1.*(segment_area < (i_segment+1))
        
        
        
        # get i-th segment and sum it up; weigh it with coefficient
        segment_area = segment_area*Isegment;
        Isegment = segment_area*xopt[i]
        # print(xopt[i])
        
        Ic = Ic + Isegment;
    return Ic
            
        
def resize_by_axis(image, dim_1, dim_2, ax, is_grayscale):
    resized_list = []
    
    
    if is_grayscale:
        unstack_img_depth_list = [tf.expand_dims(x,2) for x in tf.unstack(image, axis = ax)]
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.squeeze(tf.stack(resized_list, axis=ax))
        print(stack_img.get_shape())
    
    else:
        unstack_img_depth_list = tf.unstack(image, axis = ax)
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.stack(resized_list, axis=ax)
    
    return stack_img
    


def tf_normminmax(x):
   # x is your tensor
   
   x = x-tf.reduce_min(x)
   x = x/tf.reduce_max(x)
   return x


def tf_normmax(x):
   # x is your tensor
   

   x = x/tf.reduce_max(x)
   return x


def MeanSquareError(origImg, distImg):
    return tf.reduce_sum(tf.square(origImg - distImg))
#    
#
#def tf_fftshift(tensor):
#    ndim = len(tensor.shape)
#    for i in range(ndim):
#        n = tensor.shape[i].value
#        p2 = (n+1) // 2
#        begin1 = [0] * ndim
#        begin1[i] = p2
#        size1 = tensor.shape.as_list()
#        size1[i] = size1[i] - p2
#        begin2 = [0] * ndim
#        size2 = tensor.shape.as_list()
#        size2[i] = p2
#        t1 = tf.slice(tensor, begin1, size1)
#        t2 = tf.slice(tensor, begin2, size2)
#        tensor = tf.concat([t1, t2], axis=i)
#    return tensor
#
#
#def tf_ifftshift(tensor):
#    ndim = len(tensor.shape)
#    for i in range(ndim):
#        n = tensor.shape[i].value
#        p2 = n - (n + 1) // 2
#        begin1 = [0] * ndim
#        begin1[i] = p2
#        size1 = tensor.shape.as_list()
#        size1[i] = size1[i] - p2
#        begin2 = [0] * ndim
#        size2 = tensor.shape.as_list()
#        size2[i] = p2
#        t1 = tf.slice(tensor, begin1, size1)
#        t2 = tf.slice(tensor, begin2, size2)
#        tensor = tf.concat([t1, t2], axis=i)
#    return tensor
#

def tv_loss(x):
    #https://github.com/utkarsh2254/compression-artifacts-reduction/blob/6ebf11ff813e4bb64ab8437a56c6ffd2f99b1f7a/baseline/Losses.py
  def total_variation(images):
    pixel_dif1 = images[1:, :] - images[:-1, :]
    pixel_dif2 = images[:, 1:] - images[:, :-1]
    sum_axis = [0, 1]
    tot_var = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + \
              tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)
    return tot_var

  loss = tf.reduce_sum(total_variation(x))
  return loss


# logpath for tensorboard
logs_path = "./logs/nn_logs"
  
# variables forto save each iteration
 # initialize variables
loss_cnn = []
loss_gd = []
xopt_iter  = []
object_rotate_iter = []
t = time.time()
iter_x_opt_test = []
iter_x_opt = []
# Parameters
learning_rate = .1

display_step = 25
# num of iterations
num_steps = 500  # 272640

# how many TCC Kernels to use for simulating the image? Error already small ienough if n=2
n_kernel = 2;

matlab_data = './tf_illoptdata.mat'
object_data = './PreProcessedDataNN.mat'

with tf.device('/gpu:0'):
    
    
     ##load system data; new MATLAB v7.3 Format! 
    mat_matlab_data = h5py.File(matlab_data)
    mat_eigenfunction = np.array(mat_matlab_data['eigenfunction'])
    mat_eigenvalue = np.array(mat_matlab_data['eigenvalue'])
    mat_ill_method = np.array(mat_matlab_data['ill_method'])
        
    
    ##load input data; new MATLAB v7.3 Format! 
    mat_object_data = h5py.File(object_data)
    #mat_complxObject = np.array(mat_object_data['complxObject'])
    mat_complxObject = mat_object_data['complxObject'].value.view(np.double)
    mat_complxObject = mat_complxObject.reshape(mat_complxObject.shape[0], mat_complxObject.shape[1], mat_complxObject.shape[2]/2, 2)
    mat_complxObject = mat_complxObject[:,:,:,0] + 1j*mat_complxObject[:,:,:,1]
    
    
    # session.close()
    # tf.reset_default_graph()
    
    # determine system parameters from Matlabs eigenfunction
    n_illpattern, n_eigfct, n_system, m_system = mat_eigenfunction.shape
    
    # determine system parameters from Matlabs eigenfunction
    n_samples, n_object, m_object = mat_complxObject.shape
    
    


# logpath for tensorboard
logs_path = "/logs/nn_logs/"
filename_pb = 'expert-graph_CN.pb'
  
inputmat_name = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Python/Beamerscope_TENSORFLOW/Beamerscope_IllOpt/nninputs.mat'

outputmat_name = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Python/Beamerscope_TENSORFLOW/Beamerscope_IllOpt/nnoutputs.mat'


# session.close()
tf.reset_default_graph()


#################recreate model and safe for android###########################
    

model_filename = logs_path + filename_pb 
model_filename = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Python/Beamerscope_TENSORFLOW/Beamerscope_CNN/'+logs_path+filename_pb
    
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

# iterative over all objects, otherwise ithe GPU Memory is getting corrupted and overflows..
for object_iter in range(1040,n_samples):
        
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
        
        
    
    
    
    
    
        # Convert Matlab to Tensorflow
        tf_object_real = tf.placeholder(dtype=tf.float32, shape=[n_object, m_object])
        tf_object_imag = tf.placeholder(dtype=tf.float32, shape=[n_object, m_object])
        tf_object = tf.cast(tf.complex(tf_object_real, tf_object_imag), dtype=tf.complex64)
        tf_eigenfct = tf.constant(mat_eigenfunction, dtype=tf.float32)
        tf_eigenval = tf.constant(mat_eigenvalue, dtype=tf.float32)
        tf_xopt = tf.Variable(tf.ones([n_illpattern, 1], dtype=tf.float32))
        
        # select different initiliazation methods
        if(1):
            tf_xopt = tf.Variable(tf.random_uniform(tf_xopt.get_shape(), minval=0, maxval=1, dtype=tf.float32))
        elif(0):
            tf_xopt = 0*tf_xopt
        elif(0):
            tf_xopt = tf.ones_like(tf_xopt)    
        
        
        # Variable to store the result of the super position
        tf_I = tf.zeros([n_object, m_object], dtype=tf.float32)
        # get spectrum of the object = MATLAB: objectspectrum=fftshift(ifft2(ifftshift(object)));
        tf_object_FT_i = tf.ifft2d((tf.expand_dims(tf.expand_dims(tf_object,0),0)))
    
        
       
        # try to do it in one step
        tf_eigenfct_i = tf_eigenfct[:,0:n_kernel,:,:]
        #tf_eigenfct_i = tf.transpose(tf_eigenfct_i, [1, 2, 0])
        
       
        # convolve object's spectrum with precomputed TCC kernel from SVD 
        tf_aerial = tf.multiply(tf_object_FT_i, tf.complex(tf_eigenfct_i, tf_eigenfct_i*0))
        
        tf_aerial_FT = (tf.fft2d((tf_aerial))); # MATLAB: FTaerial=fftshift(fft2(ifftshift(aerial)));
        
        # weigh the image with the optimization factor and 
        tf_eigenval_i = tf.expand_dims(tf.expand_dims(tf_xopt*tf.square(tf_eigenval[:,0:n_kernel]), 2), 3) 
        tf_I = tf_I + tf.reduce_sum(tf_eigenval_i*tf.real(tf.multiply(tf_aerial_FT, tf.conj(tf_aerial_FT))), [0, 1])
    
        # define the cost function as the mean squared difference between the objects phase and the intensity
        
        # maximize min/max difference
        tf_min_val = tf.reduce_min(tf_I)
        tf_max_val = tf.reduce_max(tf_I)
        tf_diff_min_max = tf.abs(tf_min_val - tf_max_val)
        tf_cost = -tf_diff_min_max
            
        # minimize the error
        tf_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_cost)
            
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Launch the graph
        sess = tf.Session()
        #sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) #with tf.Session() as sess: #
        #sess = tf.Session()
        sess.run(init)
        step = 1
        
       
            
        for step in range(0,num_steps):
        
        
            object_i_real = np.real(mat_complxObject[object_iter,:,:])
            object_i_imag = np.imag(mat_complxObject[object_iter,:,:])
            
            sess.run([tf_optimizer], feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})
    
            # debug 
            # I_iter_norm, object_iter_norm, I_i, object_i = sess.run([tf_I_iter_norm, tf_object_iter_norm, tf_I, tf_object], feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})        
            
    
            # restrict optimization parameters to 0..1
            if(1):
                # cut the variables greater 1 and less than 0
                tf_xopt = tf.where(
                    tf.less(tf_xopt, tf.zeros_like(tf_xopt)),
                    tf.zeros_like(tf_xopt),
                    tf_xopt)
                
                tf_xopt = tf.where(
                    tf.greater(tf_xopt, tf.ones_like(tf_xopt)),
                    tf.ones_like(tf_xopt),
                    tf_xopt)
            elif(0):
                # push the variables into the correct range of a sigmoid 
                tf_xopt = tf.sigmoid(tf_xopt)
            else:
                # first quantize values
                tf_xopt = tf.round(tf_xopt*10)/10
                # cut the variables greater 1 and less than 0
                tf_xopt = tf.where(
                    tf.less(tf_xopt, tf.zeros_like(tf_xopt)),
                    tf.zeros_like(tf_xopt),
                    tf_xopt)
                
                tf_xopt = tf.where(
                    tf.greater(tf_xopt, tf.ones_like(tf_xopt)),
                    tf.ones_like(tf_xopt),
                    tf_xopt)
    
    
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                I_iter, loss, x_opt = sess.run([tf_I, tf_cost, tf_xopt],  feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})
                # print("Object: " + str(object_iter) + "/ " + str(n_samples) + "- Iter " + str(step) + ", Loss= " + "{:.4f}".format(loss))
    
                #plt.imshow(I_iter, cmap='gray')
    #                plt.colorbar()
    #                plt.show()
    #                    
        
    
    
        # compare the result with CNN 
        # test learned network with random spectrum
        x_test_2D = np.expand_dims(np.concatenate((np.expand_dims(object_i_real,2), np.expand_dims(object_i_imag,2)), 2), 0)
        y_pred = sess_2.run(y_store, feed_dict={x_store: x_test_2D})
        #plt.plot(y_pred.T)
        #plt.show()    
    
        sess.run([tf_optimizer], feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})
    
        I_test, loss_test = sess.run([tf_I, tf_cost],  feed_dict = {tf_xopt: y_pred.T, tf_object_real: object_i_real, tf_object_imag: object_i_imag})
     #       plt.imshow(I_test)
      #  plt.show()    
                
        
        # reset the optimization parameters for next iteration
        # tf.reset_default_graph()
        sess.run(tf_xopt, feed_dict={tf_xopt: np.random.uniform(size = tf_xopt.get_shape())})
        
    
        print("Object: " + str(object_iter) + ' result of the CNN: ' + str(loss_test) + ' and loss of GD: ' + str(loss))
        
        loss_cnn.append(loss_test)
        loss_gd.append(loss)
    #       
        iter_x_opt_test.append(y_pred)
        iter_x_opt.append(x_opt)