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

for g=149:1505;    
ct = stdfilt(img,true(33));
ct=(ct).^2;
Threshold=mean(ct(:));


for i=1:256
    for j=1:256
            if abs(C(i,j)) < (Threshold/g)
                ct(i,j) = 0;
            end 
    end
end
%g=149 #coeff=52  
%g=1505 #coeff=512

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
xlim([149 1505])
title(' Plot of MSE(Zone method) that covers 5% to 50% of coefficients')
ylabel('MSE') 
end
%Displaying Images
figure,imshowpair(uint8(img),uint8(invC),'montage');title('Original Grayscale Image (Left) and DCT Compressed Image (Right)');

