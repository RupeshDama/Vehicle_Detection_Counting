%Reading the images
road=imread('road1.png');
size_r=size(road);
carr=imread('car1.png');
size_c=size(carr);


%Background Seperation
back=ones(size_c);
back=im2uint8(back);

for i=1:size_r(1)
    for j=1:size_r(2)
        back(i,j,:)=road(i,j,:)-carr(i,j,:);
    end
end


%Converting RGB image to Grayscale
gry=ones(size_r(1),size_r(2));
gry=im2uint8(gry);

for i=1:size_r(1)
    for j=1:size_r(2)
        R=back(i,j,1);
        G=back(i,j,2);
        B=back(i,j,3);
        %Intensity values : 0.2989 * R + 0.5870 * G + 0.1140 * B 
        gry(i,j)=0.2989 * R + 0.5870 * G + 0.1140 * B;
    end
end


%Adjusting contrast
gry_adj=imadjust(gry);


%Morphological Operations: Erosion and Dilation
se = strel('square',5);
gry_erode= imerode(gry_adj,se);

se = strel('square',20);
gry_dilate= imdilate(gry_erode,se);


%Converting resulted image to Binary
gry_dilate=im2double(gry_dilate);
bin_img = imbinarize(gry_dilate,0.4);


%Blob Analysis
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true,'AreaOutputPort', true, 'CentroidOutputPort', true,'MinimumBlobArea', 150);
[area,centroid,bbox]=step(blobAnalysis, bin_img);


%Inserting shape on car
blob_img=insertShape(carr, 'Rectangle', bbox, 'Color', 'cyan','LineWidth',3);


%Representing no.of Cars
num_cars = size(bbox, 1);
res_img = insertText(blob_img,[size_r(2)-40 10], num_cars,'FontSize', 30,'BoxColor','black','BoxOpacity',0.4,'TextColor','white');


%Showing no.of Cars
figure;
subplot(1,2,1);
imshow(carr);
title('Input image');

subplot(1,2,2);
imshow(res_img);
title('Detected cars and its count');


%Showing different image processing processes
figure;
subplot(2,3,1);
imshow(carr);
title('Input image');

subplot(2,3,2);
imshow(back);
title('Background seperated');

subplot(2,3,3);
imshow(gry_adj);
title('Contrast adjusted');

subplot(2,3,4);
imshow(gry_erode);
title('Eroded')

subplot(2,3,5);
imshow(bin_img);
title('Dilated and binarized');

subplot(2,3,6);
imshow(res_img);
title('Detected cars and its count');