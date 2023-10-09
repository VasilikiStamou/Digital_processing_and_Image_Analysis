clear;clc
%Image Compression using DCT Transformation matric (T)
load flower.mat; 
img= imshow(flower);
imsave(img); 
%Read the input image
img=double(imread('flower.jpg'));

T = dctmtx(32); %Getting 32x32 transformation matrix
dct = @(block_struct)T*(block_struct.data)*T'; %Defining DCT operation
C = blockproc(img,[32 32],dct); %Finding DCT using block processing

for g=1:14.3;
ct = C;
Threshold=mean(abs(C(:)));

for i=1:256
    for j=1:256
            if abs(C(i,j)) < (Threshold/g)
                ct(i,j) = 0;
            end 
    end
end
%g=14.3 #coeff = 512  
%g=1.41 #coeff= 52

 mask= zeros(32,32);
 for i=1:32
    for j=1:32
            if ct(i,j) ~= 0
                mask(i,j) = 1;
            end 
    end
 end
m = nonzeros(mask);
%Truncating DCT coefficients 
Ct = blockproc(C,[32 32],@(block_struct) (mask .* block_struct.data));
  
invdct = @(block_struct)T' *(block_struct.data)*T'; %Defining IDCT operation
invC = blockproc(Ct,[32 32], invdct); %Finding Inverse DCT
err(g) = immse(img,invC); 
plot(err)
xlim([1.41 14.3])
title(' Plot of MSE(Threshold method) that covers 5% to 50% of coefficients')
ylabel('MSE') 
end

%Displaying Images
figure,imshowpair(uint8(img),uint8(invC),'montage');title('Original Grayscale Image (Left) and DCT Compressed Image (Right)');
