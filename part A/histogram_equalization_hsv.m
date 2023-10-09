load('barbara.mat')
img = imshow(barbara);
imsave(img); 

%Read the input image
I=imread('barbara.jpg');

%Convert the rgb image into hsv image format
HSV = rgb2hsv(I);

%Intensity Componet
v = HSV(:,:,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Perform Total Histogram Equalization on Intensity Component (HSV)
Heq_total = histeq(v);
HSV_mod_total = HSV;
HSV_mod_total(:,:,3)= Heq_total;
RGB_total = hsv2rgb(HSV_mod_total);

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
figure,imshowpair(I,RGB_total,'montage');title('Before and After Total Histogram Equalization HSV');
figure,subplot(121),bar(HIST_IN_TOTAL);colormap(mymap);legend('RED CHANNEL','GREEN CHANNEL','BLUE CHANNEL');title('Before Total Histogram Equalization HSV');
subplot(122),bar(HIST_OUT_TOTAL);colormap(mymap);legend('RED CHANNEL','GREEN CHANNEL','BLUE CHANNEL');title('After Total Histogram Equalization HSV');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Perform Local Histogram Equalization on Intensity Component (HSV)
Heq_local = adapthisteq(v);
HSV_mod_local = HSV;
HSV_mod_local(:,:,3)= Heq_local;
RGB_local = hsv2rgb(HSV_mod_local);

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
figure,imshowpair(I,RGB_local,'montage');title('Before and After Local Histogram Equalization HSV');
figure,subplot(121),bar(HIST_IN_LOCAL);colormap(mymap);legend('RED CHANNEL','GREEN CHANNEL','BLUE CHANNEL');title('Before Local Histogram Equalization HSV');
subplot(122),bar(HIST_OUT_LOCAL);colormap(mymap);legend('RED CHANNEL','GREEN CHANNEL','BLUE CHANNEL');title('After Local Histogram Equalization HSV');

figure,imshowpair(RGB_total,RGB_local,'montage');title('Total and Local HSV');











