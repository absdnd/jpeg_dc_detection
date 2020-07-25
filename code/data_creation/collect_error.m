% This file obtains the error images from the image_paths and saves it in the same location. 
function [trunc_block, round_block,dct_error_images, error_images] = collect_error(image_path)
      final_feat = [];
      error_images = [];
      dct_error_images = [];
      trunc_block = [];
      round_block = [];
      parfor img = 1:length(image_path)
            trun = 0;
            img_path = image_path{img,1};
            jpeg_img = jpeg_read(img_path);
            I = imread(img_path);

            rec = jpeg_rec_gray(jpeg_img);

            Q = jpeg_img.quant_tables{1,1};
            Q_rep = repmat(Q, size(I,1)/8, size(I,2)/8);
            
            R = double(I) - rec;
            M = int64(bdct(R)./Q_rep);
            
            err = R;
            dct_err = bdct(err);
            err = reshape(err,[1,1,size(err,1),size(err,2)]);
            dct_err = reshape(dct_err,[1,1,size(dct_err,1),size(dct_err,2)]);
            zero_8 = zeros(8,8); % For checking stability
            
      
            if(trun == 0)
                trunc_block = [trunc_block,0];
                round_block = [round_block,1];
            else
                trunc_block = [trunc_block,1];
                round_block = [round_block,0];
            end
           
            
            error_images = [error_images; err];
            dct_error_images  = [dct_error_images;dct_err];
      end
end

       
    
 
    
