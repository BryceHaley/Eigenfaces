%% last Q
% this code is from: 
%https://en.wikipedia.org/wiki/Eigenface#Matlab_example_code and
%https://www.mathworks.com/matlabcentral/answers/353060-how-to-read-multiple-pgm-images#comment_708146
%image set grabbed from ORL faces 
currentFile = mfilename( 'fullpath' );
[pathstr,~,~] = fileparts( currentFile );
addpath( fullfile( pathstr, 'orl_faces' ) );

%faceFilePath = fullfile('C:','Users','bhale','Dropbox','sfu','Fall 18',...
%    'CMPT 412','AS4','orl_faces');
faceData = imageDatastore(pathstr, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    
trainingNumFiles = 9;
rng(1) % For reproducibility

[trainFaceData,testFaceData] = splitEachLabel(faceData, ...
				trainingNumFiles,'randomize');

n = size(trainFaceData.Files);

imgs = readimage(trainFaceData,1);
for i = 2:n
    imgs = cat(3, imgs, readimage(trainFaceData,i));
end

[h,w,n] = size(imgs);
d = h*w;
% vectorize images
x = reshape(imgs,[d n]);
x = double(x);
%subtract mean
mean_matrix = mean(x,2);
x = bsxfun(@minus, x, mean_matrix);
% calculate covariance
s = cov(x');
% obtain eigenvalue & eigenvector
[V,D] = eig(s);
eigval = diag(D);
% sort eigenvalues in descending order
eigval = eigval(end:-1:1);
V = fliplr(V);
% show mean and 1th through 15th principal eigenvectors
figure,subplot(4,4,1)
imagesc(reshape(mean_matrix, [h,w]))
colormap gray
for i = 1:15
    subplot(4,4,i+1)
    imagesc(reshape(V(:,i),h,w))
end
save('eigVals.mat');
%% 
load('eigVals.mat')
clc
sizeOld = size(eigval);
eigval = eigval(1:100);
eyeguy = ones(100, 1);
zeroArr = zeros(1, sizeOld(1)-100);
eigval = cat(1, eigval,zeroArr');
eyeMat = cat(1, eyeguy, zeroArr');
eyeMat = diag(eyeMat);
D2 = diag(eigval);
V2 = V * eyeMat;

%%
clc
test = readimage(testFaceData,1);
test = reshape(test,[d,1]);
test = cast(test, 'like', mean_matrix);
resulting_test = V2' * (minus(test, mean_matrix));
disp(resulting_test(1:100));

%% 
disp('reconstructing')
clc
figure
face = (V * resulting_test) + mean_matrix;
imagesc(reshape(face, [h,w]))
colormap gray
for i = 1:10
    y{i} = x(:,i);
    y{i} = cast(y{i}, 'like', mean_matrix);
    y{i} = V2' * (minus(y{i}, mean_matrix));
end
figure
colormap gray
for i = 1:10
    face = (V * y{i}) + mean_matrix;
    subplot(2,5,i)
    imagesc(reshape(face, [h,w]))
end

%% error
clc
disp('error calculuations')
k=100;
e = zeros(1,k);
for i = 1:k
    y{i} = x(:,i);
    y{i} = cast(y{i}, 'like', mean_matrix);
    y{i} = V2' * (minus(y{i}, mean_matrix));
end

for i = 1:k
    face = (V * y{i}) + mean_matrix;
    e(1,i) = norm(face - x(:,i));
end
figure
plot(e)
title("error plot for 100 images")