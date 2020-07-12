%% The Code for training TIFS-2014
patch_size = 8;
dir_main = pwd;

% single class
ls = 0; 
% double class
ld = 1;

% list of all quality factors. 
Q_list = [20,40,60,70,75,80,85,90];


% Select the desired combination of stability index and quality factor
% Do not utilize, 20 & 40 with index = 1.  
dir_path = '../../data/';
res_path = '../../proposed_results/';
load_prefix = [dir_path, 'dataset/'];
method = 'EBSF_'
stability_index = 'all';

%Select quality factor range.
for q = 2
    
    Q_val = Q_list(q);
    
    % training dataset path.
    single_train_path  = [load_prefix, int2str(patch_size),'/train/','Quality_', int2str(Q_val),'/index_',stability_index,'/single'];    
    double_train_path  = [load_prefix, int2str(patch_size),'/train/','Quality_', int2str(Q_val),'/index_',stability_index,'/double'];

    % testing dataset path. 
    single_test_path  = [load_prefix, int2str(patch_size),'/test/','Quality_', int2str(Q_val),'/index_',stability_index,'/single'];
    double_test_path  = [load_prefix, int2str(patch_size),'/test/','Quality_', int2str(Q_val),'/index_',stability_index,'/double'];

    % result_path
    result_path =  [res_path,'EBSF', '/Quality_', int2str(Q_val), '/index_', stability_index]
   
    % load paths
    train_prefix = strcat(load_prefix, int2str(patch_size), '/train_data','/Quality_', int2str(Q_val));
    test_prefix =  strcat(load_prefix, int2str(patch_size), '/test_data', '/Quality_',int2str(Q_val));
    
    % loading training vectors. 
    single_training_new = load([single_train_path,'/' method, 'single_train']);
    double_training_new = load([double_train_path,'/',method, 'double_train']);
   
    % loading testing vectors. 
    single_testing_new = load([single_test_path,'/' method, 'single_test']);
    double_testing_new = load([double_test_path,'/',method, 'double_test']);
    
    single_training_new = single_training_new.single_vec_train;
    double_training_new = double_training_new.double_vec_train;

    single_testing_new = single_testing_new.single_vec_test;
    double_testing_new = double_testing_new.double_vec_test;
  
    % Concatenate single and double class to get entire testing dataset.
    testing_new = vertcat(single_testing_new, double_testing_new);
    testing_data = testing_new;

    % Using train size obtain single and double train. 
    train_size = size(double_training_new,1)
    itr_single_training_new = single_training_new(1:train_size,:);
    itr_double_training_new = double_training_new(1:train_size,:);
    
    training_data = vertcat(itr_single_training_new,itr_double_training_new);
  
    cd libsvm-3.21/matlab/
    
    output = test_libsvm(100,training_data,testing_data);
    cd(dir_main);
    
end
    
mkdir([res_path, 'EBSF', '/Quality_', int2str(Q_val)]);
mkdir(result_path);

%Saving output file, contains avg_accuracy which stores accuracy value. 

save([result_path,'/output_files'],'output');
cd(dir_main)

