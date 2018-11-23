%% Constants/Parameters
rng(1);
num_colors = 16;
directory = './images/baboon'; % path of directory
gray_image_path = './gray_images/baboon.png'; % path to input grayscale image
random_samples = 500;
max_images = 5;
H = 512;
W = 512;
training_images = zeros(max_images,H,W,3);
quantized_images = zeros(max_images,H,W);
mapped_quantized_images = zeros(max_images,H,W);
num_images = 0;

%% Reading all files in the directory and storing them for training
tic;
Files=dir(directory);
for k=1:length(Files)
   if(strcmp(Files(k).name(1),'.') || strcmp(Files(k).name,'..'))
       continue;
   end    
   I1 = imread(strcat(directory,'/',Files(k).name));
   I = rgb2lab(I1);
   num_images = num_images + 1;
   training_images(num_images,:,:,:) = I(:,:,:); % k-2 becasue first 2 files are . and ..
end

quantized_images(:,:,:) = training_images(:,:,:,1); % extract luminance values
           
fprintf('Images read \n');
toc;

%% Quantization of images
tic;

kmeans_arg = zeros(num_images*H*W,2); % a,b values
for i = 1:num_images
    for j = 1:H
        for k = 1:W
            kmeans_arg((i-1)*H*W + (j-1)*W + k,1) = training_images(i,j,k,2);
            kmeans_arg((i-1)*H*W + (j-1)*W + k,2) = training_images(i,j,k,3);
        end
    end
end    

[idx, C] = kmeans(kmeans_arg,num_colors,'MaxIter',10000); 

for i = 1:num_images
    for j = 1:H
        for k = 1:W
            mapped_quantized_images(i,j,k) = idx((i-1)*H*W + (j-1)*W + k,1); 
        end
    end
end   
fprintf('Quantization done \n');
toc;

%% Selection of random pixels for each image and extraction of features
tic;
feature_pixels = zeros(random_samples,2); % to store x and y 
features = zeros(num_images*random_samples, 128*3 + 441 + 2); 
bins = zeros(num_images*random_samples, 1);
for k=1:num_images
    for i=1:random_samples
        feature_pixels(i,1) = randi([1,H]);
        feature_pixels(i,2) = randi([1,W]);
    end
    feature_image = squeeze(quantized_images(k,:,:));
    % actually we need to concatenate the features into a one big matrix on which pca would be done
    temp_features = extract_feats(feature_image,feature_pixels);
    temp_bins = zeros(random_samples, 1);
    for i=1:random_samples
        temp_bins(i) = squeeze(mapped_quantized_images(k,feature_pixels(i,1),feature_pixels(i,2)));
    end
    % maintain features and bins vectors by appending temp_features and
    % temp_bins to them. temp_bins will be used by svm
    features((k-1)*random_samples+1:k*random_samples, :) = temp_features;
    bins((k-1)*random_samples+1:k*random_samples) = temp_bins;
end 

fprintf('Feature extraction done \n')
toc

%% Dimensionality reduction of features 
tic

[coeff, score1] = pca(features,'NumComponents', 32);
mu = mean(features);
mu = repmat(mu, num_images*random_samples, 1);
norm1 = features - mu;

fprintf('PCA of features done \n');
toc;

%% SVM training  
% create an array of SVMModels and also modify bins accordingly
tic

SVMModels = {};
for i=1:num_colors
    bins_temp = (bins == i); % bins_temp is a logical array now 
    SVMModels{i} = fitcsvm(score1,bins_temp,'KernelFunction','Gaussian');
end    

fprintf('SVM training done \n');

% % snippet to check output of SVMs
% for i=1:num_colors
%     cv = crossval(SVMModels{i});
%     kfoldLoss(cv)
% end

toc;

%% Prediction

grey = imread(gray_image_path);
features_grey = zeros(H*W, 128*3 + 441 + 2);
all_pixels = zeros(H*W,2); % column-major order of pixel positions

for i=1:H
    for j=1:W
       all_pixels((j-1)*H+i, 1) = i;
       all_pixels((j-1)*H+i, 2) = j;
    end    
end    

tic;
features_grey(:,:) = extract_feats(grey, all_pixels);     

mu = mean(features_grey);
mu = repmat(mu, H*W, 1);
norm2 = features_grey - mu;
features_grey_pca = norm2*coeff;

% getting svm scores
margins = zeros(num_colors, H*W);
score = {};
for i=1:num_colors
    [labels,score{i}] = predict(SVMModels{i}, features_grey_pca);
    margins(i, :) = margin(SVMModels{i}, features_grey_pca, labels);
end

% % sanity check !!!
% sane_result = zeros(H,W,3);
% for i=1:H
%     for j=1:W
%         color = -1;
%         max_till_now = -1;
%         for k = 1:num_colors
%             if(score{k}((j-1)*H+i,2) > max_till_now) 
%                 max_till_now = score{k}((j-1)*H+i,2) ;
%                 color = k;
%             end
%         end   
%         if(i < 10) 
%             disp(color);
%         end    
%         sane_result(i,j,2:3) = C(color);
%         sane_result(i,j,1) = 50;
%         if(color == -1) 
%             fprintf('error, check \n');
%         end
%     end   
% end   

% imshow(lab2rgb(sane_result));

fprintf('Feature extraction done and SVM margins obtained \n');
toc;

%% Graph Cut Optimization

% use the margins to get the costs
pairwise_costs = zeros(num_colors, num_colors);
for i=1:num_colors
    for j=1:num_colors
        pairwise_costs(i, j) = vecnorm(C(i, :) - C(j, :));
    end
end

cost_per_node = zeros(H, W, num_colors);
for i=1:num_colors
    cost_per_node(:, :, i) = -reshape(margins(i, :), [H W]);
end
[grad_mag, ~] = imgradient(imgaussfilt(grey, 3));

I2 = graph_cut(cost_per_node*100, pairwise_costs*10, grad_mag*10, grad_mag*10, 10);

fprintf('Graph cut optimization done \n');

%% Conversion to RGB
res = zeros(H,W,3);
for i=1:H
    for j=1:W
        res(i,j,1) = double(grey(i,j))/2.55;
        res(i,j,2:3) =  C(I2(i,j));
    end
end    

result = lab2rgb(res);
figure; imshow(result);
