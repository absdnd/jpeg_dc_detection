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

The file structure after saving the dataset would be as follows, 

```
/data
  /dataset
     /8
       /train
        /Quality_{Qf}
          /index_all
            /single
               /1
                 /single_error.mat
                 /single_dct_error.mat
               /2
                 ....
            /double
              /1
                 /double_error.mat
                 /double_dct_error.mat
              /2
                 ....
     
```

### Training and Inference

The main code to run our approach is MCNN_Classifier.py, which use the default runtime settings of our model as mentioned in the paper. They include, 

#### Command Line Parameters: 

These parameters can be supplied at the time of running the code. Below are the possible combinations that can be given for command line arguments. 

-  Quality factor `--Qf = {20,40,60,70,75,80,85,90}`
-  Stability index  `--index = {'1' or 'all'}`
-  Number of error images `--stack ={2,3}`
-  Run for all quality factors `--all_Q = {0,1}`
-  Number of repeated runs, `--runs = {1,2,3,4,5,6,7,8,9,10}`

#### Sample Command. 

To run at quality  Qf = 60, index = 1 and stack = 2 for 2 runs the code is: 
```shell
python /code/generate_results/MCCNN.py \
-- Qf = 60 \
-- index = 1 \
-- stack = 2\
-- runs = 2
```

The resultant directory structure, with `output_files.mat` containing the resultant predictions. 
```
/proposed_results
  /MCNN
	  /Quality_~
		   /index_1
			    /output_files.mat; 
      /index_all/
          /output_files.mat;
```






