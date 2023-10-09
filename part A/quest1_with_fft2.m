clear; clc;
image=imread('aerial.tiff');
img = uint8(255*mat2gray(image));
[row,col]=size(img);



F=fft2(double(img)); %DFT of image
Fs=fftshift(F); %Shifting spectrum to centre

H=myfilter2D('gaussianLPF',row,col,30); %Getting H(u,v)
Fsf=Fs.*H; %Filtering

fimg=ifft2(fftshift(Fsf)); %Inverse DFT
imgr=uint8(real(fimg));

subplot(221)
imshow(img); title('Input Image')

subplot(222)
imshow(log(1+abs(Fs)),[]); title('Magnitude Spectrum of Image (Log)')

subplot(223)
imshow(imgr,[]); title('Filtered Image')

subplot(224)
imshow(log(1+abs(Fsf)),[]); title('Magnitude Spectrum of Filtered Image')

figure(2)
imshow(abs(Fs),[]); title('Magnitude Spectrum of Image')