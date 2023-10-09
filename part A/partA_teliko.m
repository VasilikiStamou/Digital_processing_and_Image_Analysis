clear; clc;
%Read image
image_original = imread('chart.tiff');

%Estimate variance of input image
image_original = im2double(image_original);
var_image_original = std2(image_original)^2;

%Add white Gaussian noise , SNR=10db
SNRdB = 10;
sigma_noise = sqrt(var_image_original/10^(SNRdB/10));
noise = sigma_noise*randn(size(image_original));
image_with_noise = image_original+noise; 

%Check MSE between input image and noisy image 
err0=immse(image_original,image_with_noise);
  
% Calculate power spectral density of noise
PSD_noise = abs(fft(noise)).^2/length(noise);

%Apply AdaptiveWiener filter with PSD of noise
filterd_image_1 = wiener2(image_with_noise,[5 5],PSD_noise);

%Check MSE between input image and filterd image 
err1=immse(filterd_image_1,image_original);

%Apply AdaptiveWiener filter without PSD of noise
filterd_image_2 = wiener2(image_with_noise,[5 5]);

%Check MSE between input image and filterd image 
err2=immse(filterd_image_2,image_original);


figure,subplot(121);imshow(image_with_noise);title("Image with Gaussian Noise , SNR=10db , MMSE=" +err0);
subplot(122);imshow(image_original);title('Original Image');


figure,subplot(121);imshow(image_with_noise);title('Image with Gaussian Noise , SNR=10db');
subplot(122);imshow(filterd_image_1);title("Filterd with AdaptiveWiener (PSD of noise known),MMSE=" +err1 );

figure,subplot(121);imshow(image_with_noise);title('Image with Gaussian Noise , SNR=10db');
subplot(122);imshow(filterd_image_2);title("Filterd with AdaptiveWiener (PSD of noise is unknown),MMSE=" +err2 );