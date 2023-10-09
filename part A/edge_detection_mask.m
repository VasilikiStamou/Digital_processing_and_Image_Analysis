clear;clc; 
%1
%Read the input image
image=imread('factory.jpg');
img=rgb2gray(image);
%Roberts mask for 30% and 2%
edges_roberts_1 = edge(img,'Roberts',0.30);
edges_roberts_2 = edge(img,'Roberts',0.02);
%Sobel mask for 30% and 2%
edges_sobel_1 = edge(img,'Sobel',0.30);
edges_sobel_2 = edge(img,'Sobel',0.02);
%Display
figure,imshow(img); title('Input Image')
figure,imhist(img); title('Histogram of Input Image ')
figure,imshowpair(edges_sobel_1,edges_sobel_2,'montage');title('Sobel 30%    Sobel 2%')
figure,imshowpair(edges_roberts_1,edges_roberts_2,'montage');title('Roberts 30%    Roberts 2%')

%2 
%Roberts and Sobel for 5%
edges_sobel = edge(img,'Sobel',0.05);
edges_roberts = edge(img,'Roberts',0.05);
%Display
figure,imshowpair(edges_sobel,edges_roberts,'montage');title('Sobel 5%    Roberts 5%')


%3 
%Hough for Sobel 5%
%Compute the Hough transform of the binary image returned by edge.
[H,theta,rho] = hough(edges_sobel);
%Find the peaks in the Hough transform matrix, H, using the houghpeaks function
P = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
%Find lines in the image using the houghlines function.
lines = houghlines(edges_sobel,theta,rho,P,'FillGap',5,'MinLength',7);
%Create a plot that displays the binary image with the lines superimposed on it
figure,imshowpair(edges_sobel,img,'montage');title('Hough Sobel 5%     Input Image'),hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','white');
  % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',1,'Color','cyan');

%Hough for Roberts 5%
[H,theta,rho] = hough(edges_roberts);
P = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
lines = houghlines(edges_roberts,theta,rho,P,'FillGap',5,'MinLength',7);
figure,imshowpair(edges_roberts,img,'montage');title('Hough Roberts 5%      Input Image'),hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','white');
   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',1,'Color','cyan');


