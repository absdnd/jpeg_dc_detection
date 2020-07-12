% Code to make the entire dataset. 
addpath('dependencies/jpeg_read_toolbox');
Q_list = [20,40,60,70,75,80,85,90];

dir_path = '../../data/';
files = dir(fullfile([dir_path, 'ucid.v2'],'*.tif'));

for q = 2
   Q_val = Q_list(q)
   
   prefix = strcat(dir_path, 'Compressed_UCID_gray_full/Quality_',int2str(Q_val));
   mkdir(prefix);
   mkdir([prefix,'/single']);
   mkdir([prefix,'/double']);
   mkdir([prefix,'/triple']);
   mkdir([prefix,'/fifth']);
   mkdir([prefix,'/fourth']);
   mkdir([prefix,'/sixth']);
   
   for i = 1:length(files)
      original_image = imread([files(i).folder,'/',files(i).name]);
      original_image = rgb2gray(original_image);
      imwrite(original_image, [prefix,'/single/',int2str(i),'.jpg'],'Quality',Q_val);
      
      single_image = imread([prefix,'/single/',int2str(i),'.jpg']);
      imwrite(single_image, [prefix,'/double/',int2str(i),'.jpg'],'Quality',Q_val);
      
      double_image = imread([prefix,'/double/',int2str(i),'.jpg']);
      imwrite(double_image, [prefix,'/triple/',int2str(i),'.jpg'],'Quality',Q_val);
      
      triple_image = imread([prefix,'/triple/',int2str(i),'.jpg']);
      imwrite(triple_image, [prefix,'/fourth/',int2str(i),'.jpg'],'Quality',Q_val);
      
      fourth_image = imread([prefix,'/fourth/',int2str(i),'.jpg']);
      imwrite(fourth_image, [prefix,'/fifth/',int2str(i),'.jpg'],'Quality',Q_val);

      fifth_image = imread([prefix,'/fifth/',int2str(i),'.jpg']);
      imwrite(fifth_image, [prefix, '/sixth/',int2str(i),'.jpg'],'Quality',Q_val);
   
   end 
   
end
