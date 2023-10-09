clear;clc;
%Read input image x
x = imread('chart.tiff');
%Output image y
y=psf(x);
y=cast(y,'uint8');
%Calculate X
X=fft2(double(x)); 
Xs=fftshift(X); 
%Calculate Y
Y=fft2(double(y)); 
Ys=fftshift(Y); 
%Calculate H
Hs = Ys./ Xs;
%Calculate h
h=ifft2(fftshift(Hs)); 
h=uint8(real(h));

%Inverse Filter with Threshold
for threshold = 400:450;
for i=1:256
    for j=1:256
        if (1/(abs(Hs(i,j))) < threshold)
             invHs(i,j) = 1/Hs(i,j) ;
        elseif (1/(abs(Hs(i,j))) >= threshold) 
            invHs(i,j) = (threshold*abs(Hs(i,j)))./Hs(i,j);
        end
    end
end

approx_Xs=Ys.*invHs;

%Calculate approximation of x
approx_x=ifft2(fftshift(approx_Xs)); 
approx_x=uint8(real(approx_x));

%Calculate and Plot MMSE 
err(threshold) = immse(approx_x,x); 
plot(err)
xlabel('threshold') 
ylabel('MSE')
axis([400 450 0 0.0017])
end

err1 = immse(approx_x,x);
figure,imshow(x);title('Imput Image x');
figure,imshow(approx_x);title("Approximation of x, MSE=" + err1);
figure,imshow(log(1+abs(Hs)),[]);title('Magnitude Spectrum of PSF system (Log)');






