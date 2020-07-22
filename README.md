# Block Double compression detection using Multi-Coloumn CNN.

This repository contains the code for reproducing results presented in the paper "Double JPEG Compression Detection of Distinguishable Blocks in Images Compressed With Same Quantization Matrix" (MLSP 2020)

# Environments and Dependenices

+ tensorflow = '1.12.0' (cudatoolkit=9.0 & cudnn=7.1.2)
+ scipy = '1.4.1'
+ sklearn = '0.22.1'
+ matplotlib = '3.1.3'
+ numpy = '1.18.1'

you can install all dependencies using `pip install -r requirements.txt`

# Getting Started
### Data Generation

+ Download the [UCID dataset](https://drive.google.com/drive/folders/1AFZmvEZzHjjZJA5jMgTZKk4BuZXV3zH7?usp=sharing) containing 1338 images .TIF images and place it in the ./data folder. Also compile and store the [jpeg-read-toolbox](http://dde.binghamton.edu/download/jpeg_toolbox.zip) in the correct location. 

- Then execute `./code/data_creation/data_maker.m` to create the Compressed_UCID_gray_full dataset. 

- Execute  `./code/data_creation/patch_maker.m` , to create the training and testing 8 x 8 patches. 

- After this please execute `./code/data_creation/save_error_images.m` in order to create all the error images.


+ The file structure should look like:

```
/proposed_results
 /MCNN
    /index_1
    /index_all
        /Quality_Qf
           /MCNN_stack_~_scale_~
	      /results
                  result_itr_(iteration_number).mat  
             /loss_plots
		   loss_itr_(iteration_number).mat	
	     /weights
		   best_weights_(iteration_number).mat
```


### Usage 

The main code to run our approach is MCNN_Classifier.py, which use the default runtime settings of our model as mentioned in the paper. They include, 

a) Fixed Parameters:

+ patch_size = 8
+ scale = 1
+ l2_reg = 0.0001
+ bs = 64
+ lr = 0.001
+ save_name = 'MCNN'
+ max_epochs = 60
+ val_split = 0.2


b) Command Line Parameters: These parameters can be supplied at the time of running the code. Below are the possible combinations that can be given for command line arguments. 

-  Quality factor --Qf = {20,40,60,70,75,80,85,90}
-  Stability index  --index = {'1' or 'all'} 
-  Number of error images --stack ={2,3}
-  Run for all quality factors --all_Q = {0,1} 
-  Number of repeated runs, --runs = {1,2,3,4,5,6,7,8,9,10}

### Inference: 
 
i) please be aware that Qf = {20,40} is utilized for index = 'all' in our approach. 
ii) if all_Q = 1, then the --Qf argument is ignored.

So, if you want to run at Qf = 60, index = 1 and stack = 2 for 2 runs the code is, 

python MCCNN.py --Qf=60 --index=1 --stack=2 --runs=2


3) Results:  The results are saved in a folder called proposed results, in result_itr_(iteration_number).mat, the folder hierarchy is as shown below, 

% Obtaining the baseline results

Run ./code/generate_results/EBSF.m to obtain the baseline results of our approach. 

i) Make sure to keep the libsvm folder in the same directory.

The output_file logs the results of the approach, the avg_accuracy variable stores the resulting accuracy of the run. 
The average accuracy is also displayed after the execution of the code during runtime.






