%Berk Burak Taþdemir - 2152171
%Üzeyir Topaloðlu - 2079424

%Save the file names for each dataset to D variable.
D = ["Dataset\butterfly","Dataset\crab","Dataset\dolphin","Dataset\elephant","Dataset\flamingo","Dataset\scorpion"];
try
    delete(fullfile('db.mat'));
catch
end

%Segment each image in Dataset file.
for i = 1:length(D)
    S = dir(fullfile(D(i),'*.jpg'));
    for k = 1:length(S)
        P = fullfile(D(i),S(k).name);
        I = imread(P);
        %Blur image with average filter in order to differantiate textures.
        I = imfilter(I, ones(5)/25);
        %k-means color segmentation by 2.
        L = imsegkmeans(I,2);
        B = labeloverlay(I,L);
        %Texture segmentation.
        R = rangefilt(B,ones(11));
        %If image has 1 channel, it means it's a gray image so don't need
        %to convert to rgb2gray.
        [rows, columns, numberOfColorChannels] = size(R);
        if numberOfColorChannels > 1
            Imgray = rgb2gray(R);
        else
            Imgray = R;
        end
        %Reduce Gaussian noise by wiener filter.
        K = wiener2(Imgray,[10 10]);
        %Edge dedection using canny method.
        BW = edge(K,'canny');
        %Convert binary image.
        BW = imbinarize(double(BW));
        %Closing operation.
        BW = imclose(BW, strel('square', 4));
        %Detect all region lines using "regionprops" function by pixel
        %coordinates.
        reg=regionprops(BW,'PixelList');
        %Convert the line pixels to a vector.
        pList = cat(1,reg.PixelList);
        %Take the mean of the lines' X coordinates.
        m1=mean(pList(:,1));
        %Take the mean of the lines' Y coordinates.
        m2=mean(-pList(:,2));
        %Append to a table for feature extraction.
        F=[m1 m2];
        %load db.mat table if it's created and save it with F.
        try 
            load db;
            F=[F i];
            db=[db; F];
            save db.mat db
        catch 
            db=[F i];
            save db.mat db
        end     
    end
end

load db;

%Take db's first and second column as they are the features of the image
X = db(:,[1 2]);
%Take db's third column as it is the class of the images.
Y = db(:, 3);

%Use these features and classes in order to classify the images according
%to KNN and K-Fold cross validation.
CVKNNMdl = fitcknn(X, Y,'NumNeighbors',3,'kFold',15,'Standardize',1);

%For reproducibility
rng(1); 
%Show the classification error.
classError = kfoldLoss(CVKNNMdl)





