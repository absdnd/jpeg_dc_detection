function error = get_error_image(image_path)
  folder = image_path.folder;
  error = [];
  parfor h = 1:length(image_path)
    if(rem(h,1000)==0)
        h
    end
    read_path = strcat(folder,'/', image_path(h).name);
    jpeg_1 = jpeg_read(read_path);
    I1  = imread(read_path);
    rec1 = jpeg_rec_gray(jpeg_1); 
    err = double(I1) - rec1;
    err = reshape(err,[1,1,size(err,1),size(err,2)]);
    error = [error;err];
  end
end
    