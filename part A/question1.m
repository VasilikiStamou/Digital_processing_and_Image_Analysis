clear; clc;
%Read image
image_original = imread('clock.tiff');

%Estimate variance of input image
image_original = im2double(image_original);
var_image_original = std2(image_original)^2;

%Add white Gaussian noise , SNR=15db
SNRdB = 15;
sigma_noise = sqrt(var_image_original/10^(SNRdB/10));
noise = sigma_noise*randn(size(image_original));
image_with_noise = image_original+noise; 
err0=immse(image_original,image_with_noise);
figure,imshowpair(image_original,image_with_noise,'montage');title("Original and Noisy Image , MSE=" +err0);


%Moving Average Filter
windowSize = 3; 
kernel = ones(windowSize, windowSize) / windowSize ^ 2;
image_filtered_motion_average = imfilter(image_with_noise, kernel, 'symmetric');
err1=immse(image_filtered_motion_average,image_original);
figure,imshowpair(image_original,image_filtered_motion_average,'montage');title("Original and Moving Average Image , MSE=" +err1);

%Median Filter
image_filtered_median = medfilt2(image_with_noise);
err2=immse(image_filtered_median,image_original);
figure,imshowpair(image_original,image_filtered_median,'montage');title("Original and Median Image , MSE=" +err2);


