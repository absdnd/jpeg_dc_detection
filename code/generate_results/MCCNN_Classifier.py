### Author: Abhinav Narayan Harish ### 


# Import the Necessary Libraries.

from scipy.io import loadmat
import numpy as np
import scipy
import argparse
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tensorflow import set_random_seed 
set_random_seed(2)
import numpy as np
np.random.seed(1337)
import sys
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense,AveragePooling2D,Conv2D, Concatenate, MaxPool2D, BatchNormalization,Dropout,LeakyReLU
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape,Flatten
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from tensorflow.keras.regularizers import l2


## Using the Argparser to set default parameters. 

parser = argparse.ArgumentParser()
parser.add_argument("--Qf",default = 60, type = int)
parser.add_argument("--index", default = '1',type = str)
parser.add_argument("--stack", default = 3);
parser.add_argument('--all_Q',default = 0, type = int)
parser.add_argument('--runs', default = 1, type = int)
args = parser.parse_args()


## These parameters can be changed using argparser.


runs = args.runs
Qf = args.Qf
stability_index = args.index
k_single  = int(args.stack)
all_Q = args.all_Q

### Fixed Parameters

patch_size = 8
scale = 1
l2_reg = 0.0001
method = 'MCNN'
bs = 64
lr = 0.001
save_name = 'MCNN'
max_epochs = 60
val_split = 0.2

dir_path  = '../../data/dataset/'
res_prefix = '../../proposed_results/'

def mkdir(result_path):
      if(os.path.isdir(result_path)==False):
          os.mkdir(result_path)
      else:
          return 

# Function to obtain TPR = (True Positives/(True Positive + False Negatives))

def true_positive(y_true, y_pred):

      true_positives = np.sum(y_true*y_pred)
      possible_positives = np.sum(y_true == 1)
      return 1.0*true_positives/possible_positives

## Function to obtain confusion matrix [TP; FN; FP; TN]

def confusion_mat(y_true,y_pred):
      true_positive = np.sum((y_true==1) & (y_pred==1))
      false_positive = np.sum((y_true == 0) & (y_pred == 1))
      true_negative = np.sum((y_true == 0) & (y_pred == 0))
      false_negative = np.sum((y_true == 1) & (y_pred == 0))
      return np.array([[true_positive, false_positive],[false_negative, true_negative]])

## The true negative value is obtained using y_true & y_pred. 
def true_negative(y_true, y_pred):

      possible_negatives = np.sum(y_true == 0)
      false_negatives = np.sum(y_pred[y_true==0])
      true_negatives = possible_negatives - false_negatives
      return 1.0*true_negatives/possible_negatives


## Scaling the classifier to have mean 0 and std 1. 

def scale_train_test(train, test, k_single = 2):

    
     ret_data_train = np.zeros(np.shape(train))
     ret_data_test = np.zeros(np.shape(test))

     for i in range(k_single): 
           
           scaler = StandardScaler()
           
           error_image = train[:,:,:,i]
           error_image = error_image.reshape(len(error_image), -1)
           
           scaler.fit(error_image)
	   
           error_image = scaler.transform(error_image)
           error_image = error_image.reshape(len(error_image), 8,8)
	   
           ret_data_train[:,:,:,i] = deepcopy(error_image)


           error_image = test[:,:,:,i]  
           error_image = error_image.reshape(len(error_image),-1)

           error_image = scaler.transform(error_image)
           error_image = error_image.reshape(len(error_image), 8,8)

           ret_data_test[:,:,:,i] = deepcopy(error_image)
     
     

     return ret_data_train, ret_data_test

## Custom error metric to account for class imbalance

def custom(y_true, y_pred):
    eps = K.constant(0.00001)
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'float32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'float32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'float32'))
    tn = K.cast(K.shape(y_true)[0],'float32') - (tp + fp + fn) 
    accp = tp / (tp + fn + eps) 
    accn = tn / (tn + fp + eps)
    print(accp,accn)
    return (accp + accn)/2


## Multi Coloumn CNN network definition. 

def Multi_Coloumn_CNN(k_single):

     
     global l2_reg
     input_img = Input(shape=(8,8,k_single))
     inter = input_img
     
     enc = Conv2D(12, (3, 3), activation='linear',padding = 'same',kernel_regularizer=l2(l2_reg))(inter)
     temp1 = BatchNormalization()(enc)
     temp1  = LeakyReLU(alpha=0.2)(temp1)
     enc  = Concatenate()([temp1, inter])

     temp2 = Conv2D(12, (3, 3), activation='linear',padding = 'same',kernel_regularizer=l2(l2_reg))(enc)
     temp2  = BatchNormalization()(temp2)
     temp2  = LeakyReLU(alpha=0.2)(temp2)
          

     enc  = Concatenate()([enc, temp2, inter]) 
     enc = Conv2D(12, (1,1), activation='linear',padding = 'same', kernel_regularizer=l2(l2_reg))(enc)
     enc  = BatchNormalization()(enc)
     enc = LeakyReLU(alpha=0.2)(enc) 
          
     
     enc = AveragePooling2D(pool_size = 2)(enc)
     enc = Dropout(0.5, noise_shape=None, seed=None)(enc)
     enc = Flatten()(enc)
     
     output_1 = Dense(32, activation = 'linear',kernel_regularizer=l2(l2_reg))(enc)
     output_1 = BatchNormalization()(output_1)
     output_1 = LeakyReLU(alpha=0.2)(output_1)
     output_1 = Dropout(0.5, noise_shape=None, seed=None)(output_1)
     
     CNN = Model(input_img, output_1)
      
     return CNN

## Multi Coloumn CNN using DCT and Spatial Error.
 
def Multi_Coloumn_CNN_DCT(k_single):
     
    CNN1 = Multi_Coloumn_CNN(k_single)
    CNN2 = Multi_Coloumn_CNN(k_single)
    
    input_img_1 = Input(shape=(8,8,k_single))
    input_img_2 = Input(shape=(8,8,k_single))
    
    output_1 = CNN1(input_img_1)
    output_2 = CNN2(input_img_2)

    merged = Concatenate()([output_1, output_2])
    output = Dense(24, activation = 'linear',kernel_regularizer=l2(l2_reg))(merged)
    output = BatchNormalization()(output)
    output= LeakyReLU(alpha=0.2)(output)

    output = Dense(1, activation = 'sigmoid')(output)

    CNN1.summary()
    CNN2.summary()

    CNN = Model(inputs = [input_img_1,input_img_2],outputs = output)
    return CNN

## Choosing list of quantization table values based on stability index. 
if(all_Q==1):
	if(stability_index == 'all'):
		Q_list = [20,40,60,70,75,80,85,90]
	else:
		Q_list = [60,70,75,80,85,90]

## Reproduce for a certain Q value. 
else:
	Q_list = [Qf]



## Obtaining results for all quality factors. 
for q in range(len(Q_list)):

	Qf = Q_list[q]
	single_train_data  = []
	double_train_data  = []

	single_test_data = []
	double_test_data = []

	single_train_dct_data = []
	double_train_dct_data = []

	single_test_dct_data = []
	double_test_dct_data = []

	single_train_prefix  = dir_path + str(patch_size) + '/train/' + 'Quality_' + str(Qf) +'/index_' + str(stability_index) + '/single/'
	double_train_prefix  = dir_path + str(patch_size) + '/train/' + 'Quality_' + str(Qf) +'/index_' + str(stability_index) + '/double/'

	single_test_prefix   = dir_path + str(patch_size) + '/test/'  + 'Quality_' + str(Qf) +'/index_' + str(stability_index)+ '/single/'
	double_test_prefix   = dir_path + str(patch_size) + '/test/'  + 'Quality_' + str(Qf) +'/index_' + str(stability_index) + '/double/'


	# Loading Spatial and DCT errors. 
	for i in range(k_single):
		
		sp_load = 'error'
		dct_load = 'dct_error'
		
		single_train_error = loadmat(single_train_prefix + str(i+1) + '/single_' + str(sp_load) + '.mat')
		double_train_error = loadmat(double_train_prefix + str(i+2) + '/double_' + str(sp_load) + '.mat')
			    
		single_test_error = loadmat(single_test_prefix + str(i+1) + '/single_' + str(sp_load) + '.mat')
		double_test_error = loadmat(double_test_prefix + str(i+2) + '/double_' + str(sp_load) + '.mat')
			    
		single_train_dct = loadmat(single_train_prefix + str(i+1) + '/single_' + str(dct_load) + '.mat')
		double_train_dct = loadmat(double_train_prefix + str(i+2) + '/double_' + str(dct_load) + '.mat')

		single_test_dct  = loadmat(single_test_prefix + str(i+1) + '/single_' + str(dct_load) + '.mat')
		double_test_dct  = loadmat(double_test_prefix + str(i+2) + '/double_' + str(dct_load) + '.mat')
			    
		single_train_data.append(single_train_error['single_' + sp_load].reshape(-1,1,8,8))
		double_train_data.append(double_train_error['double_' + sp_load].reshape(-1,1,8,8))

		single_test_data.append(single_test_error['single_' + sp_load].reshape(-1,1,8,8))
		double_test_data.append(double_test_error['double_' + sp_load].reshape(-1,1,8,8))

		single_train_dct_data.append(single_train_dct['single_' + dct_load].reshape(-1,1,8,8))
		double_train_dct_data.append(double_train_dct['double_' + dct_load].reshape(-1,1,8,8))
			 
		single_test_dct_data.append(single_test_dct['single_' + dct_load].reshape(-1,1,8,8))
		double_test_dct_data.append(double_test_dct['double_' + dct_load].reshape(-1,1,8,8))
		
		    

	## Concatenating 3 error images to obtain our input tensor. 
	single_train = single_train_data[0]
	double_train = double_train_data[0]             

	single_test = single_test_data[0]
	double_test = double_test_data[0]

	single_train_dct = single_train_dct_data[0]
	double_train_dct = double_train_dct_data[0]
		
	single_test_dct  = single_test_dct_data[0]
	double_test_dct  = double_test_dct_data[0]

	
	
	for i in range(k_single-1):
		single_train = np.concatenate((single_train,single_train_data[i+1]),axis = 1)
		double_train = np.concatenate((double_train,double_train_data[i+1]),axis = 1)
		    
		single_test =  np.concatenate((single_test,single_test_data[i+1]),axis = 1)
		double_test =  np.concatenate((double_test,double_test_data[i+1]),axis = 1)

		single_train_dct = np.concatenate((single_train_dct,single_train_dct_data[i+1]),axis = 1)
		double_train_dct = np.concatenate((double_train_dct,double_train_dct_data[i+1]),axis = 1)
		     
		single_test_dct =  np.concatenate((single_test_dct ,single_test_dct_data[i+1]),axis = 1)
		double_test_dct =  np.concatenate((double_test_dct ,double_test_dct_data[i+1]),axis = 1)
		

	# Shifting the axis to obtain required input dimension (batch_size x 8 x 8 x stack) 
	single_train = np.moveaxis(single_train,1,-1)
	double_train = np.moveaxis(double_train,1,-1)

	single_test = np.moveaxis(single_test,1,-1)
	double_test = np.moveaxis(double_test,1,-1)

	single_train_dct = np.moveaxis(single_train_dct,1,-1)
	double_train_dct = np.moveaxis(double_train_dct,1,-1)
 
	single_test_dct = np.moveaxis(single_test_dct,1,-1)
	double_test_dct = np.moveaxis(double_test_dct,1,-1)

	single_train = single_train[:len(double_train),:]
	single_train_dct = single_train_dct[:len(double_train),:]

	single_labels = np.zeros([len(single_train),1],dtype='uint8')
	double_labels = np.ones([len(double_train),1],dtype='uint8')


	### Concatenating the data to obtain train and test features. 
	train = np.concatenate((single_train,double_train),axis = 0)
	train_dct = np.concatenate((single_train_dct, double_train_dct),axis = 0)

	test = np.concatenate((single_test,double_test),axis = 0)
	test_dct  = np.concatenate((single_test_dct, double_test_dct), axis = 0)

	train_labels = np.concatenate((single_labels,double_labels),axis = 0)

	# Scaling the feature to have 0 mean and 1 std.
	if(scale == 1):
		
		train, test = scale_train_test(train, test, k_single)
		train_dct, test_dct = scale_train_test(train_dct, test_dct, k_single)
		
	
	# Creating Single and Double Labels. 
	single_labels = np.zeros([len(single_test),1],dtype='uint8')
	double_labels = np.ones([len(double_test),1],dtype='uint8')

	test_labels = np.concatenate((single_labels,double_labels),axis = 0)
	
	AR_mean = 0.  
	TP_mean = 0.
	TN_mean = 0.
	
	## Splitting data into training and validation using sklearn.
	x_train, x_valid, y_train, y_valid = train_test_split(train, train_labels, test_size=val_split, shuffle= True, random_state = 42) 
	x_train_dct,x_valid_dct, y_train_dct,y_valid_dct = train_test_split(train_dct,train_labels, test_size=val_split, shuffle=True, random_state = 42)
	
	single_size = np.sum(test_labels == 0)
	double_size = np.sum(test_labels == 1)
	## Averaging results over runs. 
	for r in range(runs):

		CNN = Multi_Coloumn_CNN_DCT(k_single)
		CNN.summary()
	
		
	
		CNN.compile(optimizer = tf.keras.optimizers.Adam(lr = lr),loss='binary_crossentropy',metrics=['accuracy',custom])
		mkdir(res_prefix)
		mkdir(res_prefix + save_name)
		
		result_path = res_prefix  +  save_name + '/index_' + str(stability_index)		
		save_suffix =  method  + '_stack_' + str(k_single) +  '_scale_' + str(scale)

		# Make a directory for the result. 
		mkdir(result_path)
		result_path = result_path + '/Quality_' + str(Qf)
		mkdir(result_path)
		mkdir(result_path + '/runs/')
		mkdir(result_path + '/runs/' + save_suffix)
		mkdir(result_path + '/runs/' + save_suffix + '/weights')
		mkdir(result_path + '/runs/' + save_suffix + '/results')
		mkdir(result_path + '/runs/' + save_suffix + '/loss_plots')
					 
		filepath = result_path  + '/runs/' +  save_suffix + '/weights/best_weight_itr_' + str(r) + '.hdf5'
		checkpoint = ModelCheckpoint(filepath, monitor = 'val_custom', verbose=0, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]

		# Fit CNN with model. 
		history  = CNN.fit([x_train, x_train_dct],y_train, epochs=max_epochs, batch_size=bs, shuffle=False, validation_data = ([x_valid,x_valid_dct],y_valid),callbacks=callbacks_list) 
			
		# Saving validation plots. 
		plt.figure()
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(result_path + '/runs/' + save_suffix + '/loss_plots/acc_itr_' + str(r) + '.png')

		plt.figure()
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(result_path + '/runs/'+ save_suffix + '/loss_plots/loss_itr_' + str(r) + '.png')

		## Obtaining testing results
		CNN.load_weights(filepath)
		y_pred = CNN.predict([test,test_dct])
		
		# Rounding to closest integer to obtain labels. 

		y_pred[y_pred>=0.5] = 1
		y_pred[y_pred<=0.5] = 0
		y_test = test_labels
		
		y_pred = np.uint8(y_pred)
		TP = true_positive(y_test,y_pred)
		TN = true_negative(y_test,y_pred)
		C  = confusion_mat(y_test,y_pred)


		AR_mean += (TP+TN)/2
		TP_mean += TP
		TN_mean += TN

		#Saving individual iterations. 

		scipy.io.savemat(result_path + '/runs/' +  save_suffix + '/results/results_itr_' + str(r) + '.mat', mdict = {'AR': (TP+TN)/2, 'TP':TP, 'TN':TN , 'y_pred':y_pred, 'y_gt' : y_test,'single_size':single_size, 'double_size':double_size,'confusion':C})

	#Obtaining Single and Double Size. 	
		
	AR_mean = 1.0*(AR_mean/runs)
	TP_mean = 1.0*(TP_mean/runs)
	TN_mean = 1.0*(TN_mean/runs)


	# Saving the average results for quality factor Qf. 	
	scipy.io.savemat(result_path + '/runs/' + save_suffix + '/results/avg.mat', mdict = {'AR': AR_mean,  'TP':TP_mean, 'TN':TN_mean,'single_test_size':single_size, 'double_test_size':double_size})



