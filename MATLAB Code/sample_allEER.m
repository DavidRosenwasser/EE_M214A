%##############################################################
% Sample script to perform speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2019
%##############################################################

clear all;
clc;
%%
% Define lists
allFiles = 'allFiles.txt';
trainList = 'train_read.txt';  
testList = 'test_read.txt';

tic
%%
% Extract features
featureDict = containers.Map;   % Create features dictionary
fid = fopen(allFiles);  % Open file with directory of all audio files
myData = textscan(fid,'%s');    % Read the file
fclose(fid);    % Close the file
myFiles = myData{1};    % myData{1} contains a 565x1 cell that has all the audio file directories
for cnt = 1:length(myFiles) % For each of the 565 files
    [snd,fs] = audioread(myFiles{cnt}); % Extract sample data and sample rate
    [F0,lik] = fast_mbsc_fixedWinlen_tracking(snd,fs);  % Estimated pitch (F0) and lik = frame degree of voicing for EACH FRAME -> F0 & lik are 500x1 column vector
    featureDict(myFiles{cnt}) = mean(F0(lik>0.45)); % Finds the F0s that are most likely from voiced regions and associates with file name and then takes average
    if(mod(cnt,10)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end

%%

% Train the classifier
fid = fopen(trainList); % Opens trainList file
myData = textscan(fid,'%s %s %f');  % Read in first file, second file, whether or not speaker is the same
fclose(fid);    % Close file
fileList1 = myData{1};  % First column of trainList
fileList2 = myData{2};  % Second column of trainList
trainLabels = myData{3};    % Third column of trainList (whether or not same speaker)
trainFeatures = zeros(length(trainLabels),1);
for cnt = 1:length(trainLabels)
    trainFeatures(cnt) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt})); % Finds difference between avg F0 of first file and second file
end

Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

%%
% Test the classifier
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels),1);
for cnt = 1:length(testLabels)
    testFeatures(cnt) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

% Test the classifier
fid = fopen('test_phone.txt');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels),1);
for cnt = 1:length(testLabels)
    testFeatures(cnt) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

% Test the classifier
fid = fopen('test_mismatch.txt');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels),1);
for cnt = 1:length(testLabels)
    testFeatures(cnt) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);



% Train the classifier
fid = fopen('train_phone.txt'); % Opens trainList file
myData = textscan(fid,'%s %s %f');  % Read in first file, second file, whether or not speaker is the same
fclose(fid);    % Close file
fileList1 = myData{1};  % First column of trainList
fileList2 = myData{2};  % Second column of trainList
trainLabels = myData{3};    % Third column of trainList (whether or not same speaker)
trainFeatures = zeros(length(trainLabels),1);
for cnt = 1:length(trainLabels)
    trainFeatures(cnt) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt})); % Finds difference between avg F0 of first file and second file
end

Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

%%
% Test the classifier
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels),1);
for cnt = 1:length(testLabels)
    testFeatures(cnt) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

% Test the classifier
fid = fopen('test_phone.txt');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels),1);
for cnt = 1:length(testLabels)
    testFeatures(cnt) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

% Test the classifier
fid = fopen('test_mismatch.txt');
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
testLabels = myData{3};
testFeatures = zeros(length(testLabels),1);
for cnt = 1:length(testLabels)
    testFeatures(cnt) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);

toc
%%