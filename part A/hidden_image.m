clear;clc;
load('circle.mat')
img = imshow(circle);
imsave(img);

%Read the input image
I=imread('circle.jpg');
Img = I;

%Window size
M=11;
N=11;
mid_val=round((M*N)/2);

%Find the number of rows and columns to be padded with zero
in=0;
for i=1:M
    for j=1:N
        in=in+1;
        if(in==mid_val)
            PadM=i-1;
            PadN=j-1;
            break;
        end
    end
end

%Padding the intensity componet with zero on all sides
B=padarray(I,[PadM,PadN]);

for i= 1:size(B,1)-((PadM*2)+1)
    
    for j=1:size(B,2)-((PadN*2)+1)
        cdf=zeros(256,1);
        inc=1;
        for x=1:M
            for y=1:N
                
  %Fing the middle element in the window                      
                if(inc==mid_val)
                    ele=B(i+x-1,j+y-1)+1;
                end
                    pos=B(i+x-1,j+y-1)+1;
                    cdf(pos)=cdf(pos)+1;
                   inc=inc+1;
            end
        end
        
        %Compute the CDF for the values in the window              
        for l=2:256
            cdf(l)=cdf(l)+cdf(l-1);
        end
            Img(i,j)=round(cdf(ele)/(M*N)*255);
     end
end

J = imadjust(Img,[0.7 1],[]);
J = imsharpen(J);

%Display images before and after histogram equalization
figure,subplot(231),imshow(I);title('Before Histogram Equalization');
subplot(232),imshow(Img);title('After Histogram Equalization');
subplot(233),imshow(J);title('After Adjust and Sarpen');
subplot(234);imhist(I);ylim([0 3000]);title('Before Local Histogram Equalization'); 
subplot(235); imhist(Img);ylim([0 1500]);title('After Local Histogram Equalization');
subplot(236); imhist(J);ylim([0 60]);title('After Adjust and Sarpen');
