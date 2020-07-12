function [Y] = jpeg_rec_gray(image)
% Return the grayscale image(float, without RT) from the DCT coefs
    Y = ibdct(dequantize(image.coef_arrays{1}, image.quant_tables{1}));
    Y = Y + 128;
return
