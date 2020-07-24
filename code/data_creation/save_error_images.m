%% This code is used to save the error images of our approach.
addpath('dependencies/jpeg_read_toolbox/')
addpath('all_needed_matlab_functions/')

dir_path = '../../data/'
Q_list = [20,40,60,70,75,80,85,90]; %All quality factors reported. 
stability_index = 'all';

if(strcmp(stability_index, '1'))
    Q_list = [60,70,75,80,85,90]  %Choosing only 60-90 for Qf == 1 analysis. 
end

ld = 1;
ls = 0;
prefixes = {'test/','train/'};
patch_size = 8;

% Choose quality factors. 
for q = 1:length(Q_list)
    % Saving the error image. 
    Q_val = Q_list(q) 
    single_train_path  = [dir_path, 'dataset/', int2str(patch_size),'/train/', 'Quality_', int2str(Q_val),'/index_',stability_index,'/single'];
    single_test_path  = [dir_path,'dataset/', int2str(patch_size),'/test/', 'Quality_', int2str(Q_val),'/index_',stability_index,'/single'];
    double_train_path = [dir_path,'dataset/', int2str(patch_size),'/train/', 'Quality_', int2str(Q_val),'/index_',stability_index,'/double'];
    double_test_path = [dir_path, 'dataset/', int2str(patch_size),'/test/', 'Quality_', int2str(Q_val),'/index_',stability_index,'/double'];
    prefixes = {'test/','train/'};
    
    for m = 1:2 
        prefix = prefixes{m};
        load_prefix = [dir_path, 'patches_',prefix] 

        read_single_path  = [load_prefix, int2str(patch_size),'/Quality_', int2str(Q_val),'/index_',stability_index,'/single'];
        read_double_path  = [load_prefix, int2str(patch_size),'/Quality_', int2str(Q_val),'/index_',stability_index,'/double'];

        single_save_path  = [dir_path, 'dataset/', int2str(patch_size),'/',prefix,'Quality_', int2str(Q_val),'/index_',stability_index,'/single'];
        double_save_path  = [dir_path, 'dataset/', int2str(patch_size),'/',prefix,'Quality_', int2str(Q_val),'/index_',stability_index,'/double'];

        mkdir(single_save_path);
        mkdir(double_save_path);
        
        for k = 1:4
            mkdir([single_save_path,'/', int2str(k)]);
            mkdir([double_save_path,'/', int2str(k)]);

        end

        for k = 1:3
            
            single_path = create_img_struct(fullfile([read_single_path ,'/' ,int2str(k)]));
            double_path = create_img_struct(fullfile([read_double_path ,'/' ,int2str(k+1)]));
            
            if(strcmp(prefix,'train/'))

               if(length(single_path)>0)
                   
                     [trunc,round,single_dct_error, single_error, single_EBSF] = TIFS_2014(single_path);
               end
                
                save([single_save_path,'/',int2str(k),'/','single_error'],'single_error','trunc','round');
                save([single_save_path,'/',int2str(k),'/','single_dct_error'],'single_dct_error','trunc','round');
                
                if(k==1)
                    single_vec_train = single_EBSF;
                    single_trunc_train = trunc;
                    single_round_train = round;
                end
                
                if(length(double_path)>0)
                   
                    [trunc, round,double_dct_error, double_error, double_EBSF] = TIFS_2014(double_path);

                end
                
                save([double_save_path,'/',int2str(k+1),'/','double_error'],'double_error','trunc','round');
                save([double_save_path,'/',int2str(k+1),'/','double_dct_error'],'double_dct_error','trunc','round');
                
                if(k == 1)
                    double_vec_train = double_EBSF;
                    double_trunc_train = trunc;
                    double_round_train = round;
                end
                
            else
                
                if(length(single_path)>0)
                    [trunc,round,single_dct_error,single_error, single_EBSF] = TIFS_2014(single_path);
                end
               
                if(k==1)
                    single_vec_test = single_EBSF;
                    single_trunc_train = trunc;
                    double_round_train = round;
                end
                
               
                save([single_save_path,'/',int2str(k),'/','single_error'],'single_error','trunc','round');
                save([single_save_path,'/',int2str(k),'/','single_dct_error'],'single_dct_error','trunc','round');
                
                if(length(double_path)>0)
                   [trunc, round, double_dct_error, double_error, double_EBSF] = TIFS_2014(double_path);

                end
                
                if(k==1)
                    double_trunc_test = trunc;
                    double_round_test = round;
                    double_vec_test = double_EBSF;
                end
                
                save([double_save_path,'/',int2str(k+1),'/','double_error'],'double_error','trunc','round');
                save([double_save_path,'/',int2str(k+1),'/','double_dct_error'],'double_dct_error','trunc','round');
                
                
                
                
                
            end
        end
    end
    
    single_vec_test(:,end+1) = ls*ones(size(single_vec_test,1),1);
    single_vec_train(:,end+1) = ls*ones(size(single_vec_train,1),1);
    
    double_vec_test(:,end+1) = ld*ones(size(double_vec_test,1),1);
    double_vec_train(:,end+1) = ld*ones(size(double_vec_train,1),1);

    single_train_size = size(single_vec_train,1);
    single_test_size = size(single_vec_test,1);

    training = vertcat(single_vec_train,double_vec_train);
    testing = vertcat(single_vec_test,double_vec_test);

    TF = isnan(training);
    idx = TF == 1;
    training(idx) = 0;

    TF = isnan(testing);
    idx = TF == 1;
    testing(idx) = 0;
    
    [training_new, testing_new,indices] = remove_zero_feature(training,testing);
    single_vec_train = training_new(1:single_train_size,:);
    double_vec_train = training_new(single_train_size+1:end,:);
    
    single_vec_test = testing_new(1:single_test_size,:);
    double_vec_test = testing_new(single_test_size+1:end,:);

    save(strcat(single_train_path,'/EBSF_single_train.mat'),'single_vec_train','indices');
    save(strcat(double_train_path,'/EBSF_double_train.mat'),'double_vec_train','indices');
    save(strcat(single_test_path,'/EBSF_single_test.mat'),'single_vec_test','indices');
    save(strcat(double_test_path,'/EBSF_double_test.mat'),'double_vec_test','indices'); 
    
end

