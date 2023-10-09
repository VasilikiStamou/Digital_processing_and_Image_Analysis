clear; clc;
%Read image
image_original = imread('clock.tiff');

%Add Salt & Pepper Noise
image_with_noise = imnoise(image_original,'salt & pepper',0.2);

err0=immse(image_original,image_with_noise);
figure,imshowpair(image_original,image_with_noise,'montage');title("Original and Noisy Image , MSE=" +err0);


%Moving Average Filter
windowSize = 5; 
kernel = ones(windowSize, windowSize) / windowSize ^ 2;
image_filtered_motion_average = imfilter(image_with_noise, kernel, 'symmetric');

err1=immse(image_filtered_motion_average,image_original);
figure,imshowpair(image_original,image_filtered_motion_average,'montage');title("Original and Moving Average Image , MSE=" +err1);

%Median Filter
image_filtered_median = medfilt2(image_with_noise,[5 5]);
err2=immse(image_filtered_median,image_original);
figure,imshowpair(image_original,image_filtered_median,'montage');title("Original and Median Image , MSE=" +err2);



