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
                   
                     [trunc,round,single_dct_error, single_error] = collect_error(single_path);
               end
                
                save([single_save_path,'/',int2str(k),'/','single_error'],'single_error','trunc','round');
                save([single_save_path,'/',int2str(k),'/','single_dct_error'],'single_dct_error','trunc','round');
                
                if(k==1)
                    single_trunc_train = trunc;
                    single_round_train = round;
                end
                
                if(length(double_path)>0)
                   
                    [trunc, round,double_dct_error, double_error] = collect_error(double_path);

                end
                
                save([double_save_path,'/',int2str(k+1),'/','double_error'],'double_error','trunc','round');
                save([double_save_path,'/',int2str(k+1),'/','double_dct_error'],'double_dct_error','trunc','round');
                
                if(k == 1)
                    double_trunc_train = trunc;
                    double_round_train = round;
                end
                
            else
                
                if(length(single_path)>0)
                    [trunc,round,single_dct_error,single_error] = collect_error(single_path);
                end
               
                if(k==1)
                    single_trunc_train = trunc;
                    double_round_train = round;
                end
                
               
                save([single_save_path,'/',int2str(k),'/','single_error'],'single_error','trunc','round');
                save([single_save_path,'/',int2str(k),'/','single_dct_error'],'single_dct_error','trunc','round');
                
                if(length(double_path)>0)
                   [trunc, round, double_dct_error, double_error] = collect_error(double_path);

                end
                
                if(k==1)
                    double_trunc_test = trunc;
                    double_round_test = round;
                end
                
                save([double_save_path,'/',int2str(k+1),'/','double_error'],'double_error','trunc','round');
                save([double_save_path,'/',int2str(k+1),'/','double_dct_error'],'double_dct_error','trunc','round');
                
                
                
                
                
            end
        end
    end
    
    
end

