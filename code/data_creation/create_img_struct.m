function [img_struct] = create_img_struct(dir_path)
    addpath('dependencies/jpeg_read_toolbox');
    
    if(nargin == 1)
        image_struct = dir(fullfile(dir_path,'*.jpg'));
        array_size = length(image_struct)
        used_size = length(image_struct);
        seed = 2;
   
    end
    rand('seed',seed);
    choices = randperm(array_size);
    choices = choices(1:used_size);
    img_struct = {};
    for i = 1:length(choices)
        ch = choices(i);
        read_path = [dir_path, '/' ,int2str(ch), '.jpg'];
        img_struct{i,1} = read_path;
    end
end
   