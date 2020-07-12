% This file obtains the error images from the image_paths and saves it in the same location. 
function [trunc_block, round_block,dct_error_images, error_images, final_feat] = TIFS_2014(image_path)
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
            
            r_error = [];
            r_error_dc = [];
            r_error_ac = [];
            t_error = [];
            t_error_dc = [];
            t_error_ac = [];

            for i = 1:8:size(M,1)
                for j = 1:8:size(M,2)
                    M_n = M(i:i+7, j:j+7);
                    % Process unstable block only
                    if (nnz(M_n == zero_8) ~= 64)
                        R_n = R(i:i+7, j:j+7);
                        W_n = double(M_n).*Q; % Dequantized DCT of the error block
                        W_n = reshape(W_n, 1, 64);

                        % Rounding error block
                        if(max(max(R_n)) <= 0.5 && min(min(R_n)) >= -0.5)
                            trun = 0;
                            r_error = [r_error, R_n];
                            r_error_dc = [r_error_dc, W_n(1)]; % DC comp
                            r_error_ac = [r_error_ac, W_n(2:end)]; % AC comp

                        % Truncation error block
                        else
                            trun = 1;
                            t_error = [t_error, R_n];
                            t_error_dc = [t_error_dc, W_n(1)]; % DC comp
                            t_error_ac = [t_error_ac, W_n(2:end)]; % AC comp
                        end            
                    end
                end
            end
      
            if(trun == 0)
                trunc_block = [trunc_block,0];
                round_block = [round_block,1];
            else
                trunc_block = [trunc_block,1];
                round_block = [round_block,0];
            end
            % Make the final feature vector
            feature_vec = [mean2(abs(r_error)), var(abs(r_error(:))), mean2(abs(t_error)), var(abs(t_error(:))), mean(abs(r_error_dc)), var(abs(r_error_dc)), mean(abs(r_error_ac)), var(abs(r_error_ac)), mean(abs(t_error_dc)), var(abs(t_error_dc)), mean(abs(t_error_ac)), var(abs(t_error_ac)), length(r_error_dc)/(length(r_error_dc)+length(t_error_dc))];
            final_feat = [final_feat;feature_vec];
            error_images = [error_images; err];
            dct_error_images  = [dct_error_images;dct_err];
      end
end

       
    
 
    
