image=imread('aerial.tiff');
img = uint8(255*mat2gray(image));

Fimg=abs(fftshift(fft2(img)));
figure
subplot(121)
imshow(img); title('Spatial domain')
subplot(122)
imshow(log(1+Fimg),[]); title('Frequency domain')

M=256; N=256; D0=20; n=1;
HidealLPF = myfilter2D('idealLPF',M,N,D0);
HbutterLPF = myfilter2D('butterLPF',M,N,D0,n);
HgaussianLPF = myfilter2D('gaussioanLPF',M,N,D0);
figure 
subplot(231)
imshow(HidealLPF,[]); title('Ideal Low Pass Filter') 
subplot(232)
imshow(HbutterLPF,[]); title('Butterworth Low Pass Filter') 
subplot(233)
imshow(HgaussianLPF,[]); title('Gaussian Low Pass Filter')
subplot(234)
mesh(HidealLPF); title('Ideal Low Pass Filter (3D)')
subplot(235)
mesh(HbutterLPF); title('Butterworth Low Pass Filter (3D)') 
subplot(236)
mesh(HgaussianLPF); title('Gaussian Low Pass Filter (3D)')
