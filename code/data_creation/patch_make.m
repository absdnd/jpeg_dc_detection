
addpath('dependencies/jpeg_read_toolbox');

Q_list = [20,40,60,70,75,80,85,90];

dir_path = '../../data/'
files = dir(fullfile([dir_path, 'ucid.v2'],'*.tif'));
patch_size_1 = 8;
patch_size_2 = 8;

stability_index = 'all'

train = true;

if(train)
    save_prefix = [dir_path, 'patches_train/'];
    image_range = 1:length(files)/2;
else
    save_prefix = [dir_path, 'patches_test/'];
    image_range = length(files)/2+1:length(files);
end


for q = 1:length(Q_list)

   Q_val = Q_list(q)

   prefix = strcat(dir_path, 'Compressed_UCID_gray_full/Quality_',int2str(Q_val));
   
   mkdir([save_prefix, int2str(patch_size_1) filesep 'Quality_' int2str(Q_val) filesep 'index_',stability_index]);
   mkdir([save_prefix, int2str(patch_size_1) filesep 'Quality_' int2str(Q_val) filesep 'all']);
   mkdir([save_prefix, int2str(patch_size_1) filesep 'Quality_' int2str(Q_val) filesep 'index_',stability_index filesep 'single']);
   mkdir([save_prefix, int2str(patch_size_1) filesep 'Quality_' int2str(Q_val) filesep 'index_',stability_index filesep 'double']);
   
   for k = 1:5
     mkdir([save_prefix, int2str(patch_size_1) filesep 'Quality_' int2str(Q_val) filesep 'index_',stability_index filesep 'double' filesep int2str(k)]);
     mkdir([save_prefix, int2str(patch_size_1) filesep 'Quality_' int2str(Q_val) filesep 'index_',stability_index filesep 'single' filesep int2str(k)]);
   end
   single_path  = strcat(prefix,'/single/');
   double_path  = strcat(prefix,'/double/');
   triple_path  = strcat(prefix,'/triple/');
   fourth_path  = strcat(prefix,'/fourth/');
   
   write_prefix = strcat(save_prefix, int2str(patch_size_1),'/Quality_', int2str(Q_val));

   cnt = 0
   cnt_single = 0; 
   cnt_double = 0; 
   diff1 = 0;
   diff2 = 0;
   diff3 = 0;
   for f = image_range
        f
        single_image = jpeg_read(strcat(single_path,int2str(f),'.jpg'));
        double_image = jpeg_read(strcat(double_path,int2str(f),'.jpg'));
        triple_image = jpeg_read(strcat(triple_path,int2str(f),'.jpg'));
        fourth_image = jpeg_read(strcat(fourth_path,int2str(f),'.jpg'));
        
        rows = size(single_image.coef_arrays{1},1);
        cols = size(single_image.coef_arrays{1},2);

        for i = 1:patch_size_1:rows-patch_size_1 + 1
            for j = 1:patch_size_2:cols-patch_size_2 + 1
             
                original_image = imread(strcat(files(f).folder,'/',files(f).name));
                original_image = rgb2gray(original_image);
               
                if(strcmp(stability_index, '1'))
                    diff1 = nnz(single_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1) - double_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1));
                    diff2 = nnz(double_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1) - triple_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1));
                    diff3 = nnz(triple_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1) - fourth_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1));
                
                elseif(strcmp(stability_index, 'all'))
                    diff1 = nnz(single_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1) - double_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1));
                    diff2 = nnz(double_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1) - triple_image.coef_arrays{1}(i:i+patch_size_1-1,j:j+patch_size_2-1));
                    diff3 = -1;
                    
                else
                    
                   error('Incorrect Stability Index value, use: "1" or "all"')
                    
                end
                
                if(diff1 > 0 & diff2 == 0 & diff3~=-1)
                    
                    cnt_single = cnt_single + 1;
                    imwrite(original_image(i:i+patch_size_1-1,j:j+patch_size_2-1),strcat(write_prefix,'/index_',stability_index,'/single/',int2str(1),'/',int2str(cnt_single),'.jpg'),'Quality',Q_val);
                    jpeg_single_image = imread(strcat(write_prefix,'/index_',stability_index,'/single/',int2str(1),'/',int2str(cnt_single),'.jpg'));
                    imwrite(jpeg_single_image, strcat(write_prefix,'/index_',stability_index,'/single/',int2str(2),'/',int2str(cnt_single),'.jpg'),'Quality',Q_val);
                    
                    jpeg_double_image = imread(strcat(write_prefix,'/index_',stability_index,'/single/',int2str(2),'/',int2str(cnt_single),'.jpg'));
                    imwrite(jpeg_double_image, strcat(write_prefix,'/index_',stability_index,'/single/',int2str(3),'/',int2str(cnt_single),'.jpg'),'Quality',Q_val);
                    
                    jpeg_triple_image = imread(strcat(write_prefix,'/index_',stability_index,'/single/',int2str(3),'/',int2str(cnt_single),'.jpg'));
                    imwrite(jpeg_triple_image, strcat(write_prefix,'/index_',stability_index,'/single/',int2str(4),'/',int2str(cnt_single),'.jpg'),'Quality',Q_val);
                end
                
               
                
                if(diff2 > 0 & diff3 == 0)
                    cnt_double = cnt_double + 1;
                    imwrite(original_image(i:i+patch_size_1-1,j:j+patch_size_2-1),strcat(write_prefix,'/index_',stability_index,'/double/',int2str(1),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);
                    jpeg_single_image = imread(strcat(write_prefix,'/index_',stability_index,'/double/',int2str(1),'/',int2str(cnt_double),'.jpg'));
                    imwrite(jpeg_single_image, strcat(write_prefix,'/index_',stability_index,'/double/',int2str(2),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);
                  
                    jpeg_double_image = imread(strcat(write_prefix,'/index_',stability_index,'/double/',int2str(2),'/',int2str(cnt_double),'.jpg'));
                    imwrite(jpeg_double_image, strcat(write_prefix,'/index_',stability_index,'/double/',int2str(3),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);
                    
                    jpeg_triple_image = imread(strcat(write_prefix,'/index_',stability_index,'/double/',int2str(3),'/',int2str(cnt_double),'.jpg'));
                    imwrite(jpeg_triple_image, strcat(write_prefix,'/index_',stability_index,'/double/',int2str(4),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);
                    
                    jpeg_fourth_image = imread(strcat(write_prefix,'/index_',stability_index,'/double/',int2str(4),'/',int2str(cnt_double),'.jpg'));
                    imwrite(jpeg_fourth_image, strcat(write_prefix,'/index_',stability_index,'/double/',int2str(5),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);

                end
                
                
                if(diff1 > 0 & diff3 == -1)
                    
                    cnt_single = cnt_single + 1;
                    imwrite(original_image(i:i+patch_size_1-1,j:j+patch_size_2-1),strcat(write_prefix,'/index_all','/single/',int2str(1),'/',int2str(cnt_single),'.jpg'),'Quality',Q_val);
                    jpeg_single_image = imread(strcat(write_prefix,'/index_all','/single/',int2str(1),'/',int2str(cnt_single),'.jpg'));
                    imwrite(jpeg_single_image, strcat(write_prefix,'/index_all','/single/',int2str(2),'/',int2str(cnt_single),'.jpg'),'Quality',Q_val);
                    
                    jpeg_double_image = imread(strcat(write_prefix,'/index_all','/single/',int2str(2),'/',int2str(cnt_single),'.jpg'));
                    imwrite(jpeg_double_image, strcat(write_prefix,'/index_all','/single/',int2str(3),'/',int2str(cnt_single),'.jpg'),'Quality',Q_val);
                
                end
                
                if(diff2 > 0 & diff3 == -1)
                    cnt_double = cnt_double + 1;
                    imwrite(original_image(i:i+patch_size_1-1,j:j+patch_size_2-1),strcat(write_prefix,'/index_all','/double/',int2str(1),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);
                    jpeg_single_image = imread(strcat(write_prefix,'/index_all','/double/',int2str(1),'/',int2str(cnt_double),'.jpg'));
                    imwrite(jpeg_single_image, strcat(write_prefix,'/index_all','/double/',int2str(2),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);
                  
                    jpeg_double_image = imread(strcat(write_prefix,'/index_all','/double/',int2str(2),'/',int2str(cnt_double),'.jpg'));
                    imwrite(jpeg_double_image, strcat(write_prefix,'/index_all','/double/',int2str(3),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);
                    
                    jpeg_triple_image = imread(strcat(write_prefix,'/index_all','/double/',int2str(3),'/',int2str(cnt_double),'.jpg'));
                    imwrite(jpeg_triple_image, strcat(write_prefix,'/index_all','/double/',int2str(4),'/',int2str(cnt_double),'.jpg'),'Quality',Q_val);
               
               
                end
                       
            end
                
        end
     end
 end
