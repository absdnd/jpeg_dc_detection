# Block level double compression detection using Multi-Coloumn CNN.

This repository contains source code useful for reproducing results presented in the paper "Double JPEG Compression Detection of Distinguishable Blocks in Images Compressed With Same Quantization Matrix" (MLSP 2020)

# Usage

1) Fixed Parameters: These have been defined within the MCCNN_Classifier.py, are a part of the default runtime settings of our model as mentioned in the paper. 

They include, 
\begin{itemize}
\item patch_size = 8
\item scale = 1
\item l2_reg = 0.0001
\item method = 'MCNN'
\item bs = 64
\item lr = 0.001
\item save_name = 'MCNN'
\item max_epochs = 60
\item val_split = 0.2



2) Command Line Parameters: These parametera can be supplied at the time of running the code. Below are the possible combinations that can be given for command line arguments. 

i)   Quality factor --Qf = {20,40,60,70,75,80,85,90}
ii)  Stability index  --index = {'1' or 'all'} 
iii) Number of error images --stack ={2,3}
(iv) Run for all quality factors --all_Q = {0,1} 
(v)  --runs = {1,2,3,4,5,6,7,8,9,10}

# Note : 
 
i) please be aware that Qf = {20,40} is utilized for index = 'all' in our approach. 
ii) if all_Q = 1, then the --Qf argument is ignored.

So, if you want to run at Qf = 60, index = 1 and stack = 2 for 2 runs the code is, 

python MCCNN.py --Qf=60 --index=1 --stack=2 --runs=2


3) Results:  The results are saved in a folder called proposed results, in result_itr_(iteration_number).mat, the folder hierarchy is as shown below, 


proposed_results
 |
 |    
 +-- MCNN
 	| 
	+--index_1
	|
	+--index_all
		|
		+--Quality_Qf
			|
			+--MCNN_stack_~_scale_~
				|
				+--results
				|	|
				|	+--result_itr_(iteration_number).mat
				|	
				+--loss_plots
				|	|
				|	+--loss_itr_(iteration_number).mat	
				|	
				+--weights
					|
					+--best_weights_(iteration_number).mat




% Obtaining the baseline results:

Run ./code/generate_results/EBSF.m to obtain the baseline results of our approach. 

i) Make sure to keep the libsvm folder in the same directory.
ii) Please use Qf = {20,40} only with 'index=all' 


1) Results: The results are saved in the proposed_results folder in the following format:


proposed_results
 |
 |
 +---EBSF
	|
	+--Quality_Qf
		|
		+--index_stability_~
			|
			output_files.mat; 


The output_file logs the results of the approach, the avg_accuracy variable stores the resulting accuracy of the run. 
The average accuracy is also displayed after the execution of the code during runtime.



%%% Instructions for Creating the Data %%%% 



1. Download the UCID dataset containing 1338 images .TIF images and place it in the ./data folder. 

2. Then execute ./code/data_creation/data_maker.m to create the Compressed_UCID_gray_full dataset. 

3. Execute  ./code/data_creation/patch_maker.m , to create the training and testing 8 x 8 patches. 

4. After this please execute ./code/data_creation/save_error_images.m in order to create all the error images.

5. This should create the data for the training and testing pipeline. 




% Important Note: Please be aware that this saves all 8x8 possible blocks in our dataset and hence will take a long time to execute and utilize a lot of fragmented memory.
We have created a sample dataset for Qf = 20 and index = 'all' for usage. The error images for other Qf's are already saved. 
