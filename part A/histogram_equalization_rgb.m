load('barbara.mat')
img = imshow(barbara);
imsave(img); 
%Read the input image
I=imread('barbara.jpg');
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Perform Total Histogram Equalization each component (RGB)
Req = histeq(R);
Geq = histeq(G);
Beq = histeq(B);
RGB_total = I;
RGB_total(:,:,1) = Req;
RGB_total(:,:,2) = Geq;
RGB_total(:,:,3) = Beq;

%Display the histogram of the original and the equalized Image
HIST_IN_TOTAL = zeros([256 3]);
HIST_OUT_TOTAL = zeros([256 3]);

%Histogram of the RED,GREEN and BLUE Components
HIST_IN_TOTAL(:,1) = imhist(I(:,:,1),256); %RED
HIST_IN_TOTAL(:,2) = imhist(I(:,:,2),256); %GREEN
HIST_IN_TOTAL(:,3) = imhist(I(:,:,3),256); %BLUE
HIST_OUT_TOTAL(:,1) = imhist(RGB_total(:,:,1),256); %RED
HIST_OUT_TOTAL(:,2) = imhist(RGB_total(:,:,2),256); %GREEN
HIST_OUT_TOTAL(:,3) = imhist(RGB_total(:,:,3),256); %BLUE
mymap=[1 0 0; 0.2 1 0; 0 0.2 1];

%Display images before and after histogram equalization
figure,imshowpair(I,RGB_total,'montage');title('Before and After Total Histogram Equalization RGB');
figure,subplot(121),bar(HIST_IN_TOTAL);colormap(mymap);legend('RED CHANNEL','GREEN CHANNEL','BLUE CHANNEL');title('Before Total Histogram Equalization RGB');
subplot(122),bar(HIST_OUT_TOTAL);colormap(mymap);legend('RED CHANNEL','GREEN CHANNEL','BLUE CHANNEL');title('After Total Histogram Equalization RGB');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Perform Local Histogram Equalization on each component (RGB)

Req = adapthisteq(R);
Geq = adapthisteq(G);
Beq = adapthisteq(B);
RGB_local = I;
RGB_local(:,:,1) = Req;
RGB_local(:,:,2) = Geq;
RGB_local(:,:,3) = Beq;

%Display the histogram of the original and the equalized Image
HIST_IN_LOCAL = zeros([256 3]);
HIST_OUT_LOCAL = zeros([256 3]);

%Histogram of the RED,GREEN and BLUE Components
HIST_IN_LOCAL(:,1) = imhist(I(:,:,1),256); %RED
HIST_IN_LOCAL(:,2) = imhist(I(:,:,2),256); %GREEN
HIST_IN_LOCAL(:,3) = imhist(I(:,:,3),256); %BLUE
HIST_OUT_LOCAL(:,1) = imhist(RGB_local(:,:,1),256); %RED
HIST_OUT_LOCAL(:,2) = imhist(RGB_local(:,:,2),256); %GREEN
HIST_OUT_LOCAL(:,3) = imhist(RGB_local(:,:,3),256); %BLUE
mymap=[1 0 0; 0.2 1 0; 0 0.2 1];

%Display images before and after histogram equalization
figure,imshowpair(I,RGB_local,'montage');title('Before and After Local Histogram Equalization RGB');
figure,subplot(121),bar(HIST_IN_LOCAL);colormap(mymap);legend('RED CHANNEL','GREEN CHANNEL','BLUE CHANNEL');title('Before Local Histogram Equalization RGB');
subplot(122),bar(HIST_OUT_LOCAL);colormap(mymap);legend('RED CHANNEL','GREEN CHANNEL','BLUE CHANNEL');title('After Local Histogram Equalization RGB');

figure,imshowpair(RGB_total,RGB_local,'montage');title('Total and Local RGB');

