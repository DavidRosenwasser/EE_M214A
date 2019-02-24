%##############################################################
% Script to perform speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2019
% Hamza Zamani, David Rosenwasser, & Taishi Kato
%##############################################################

clear variables;
clc;
%%
% Definitions
allFiles = 'allFiles.txt';
EER_Matrix = zeros(2,3);
MFCC_Num = 12;

tic
%% 
% Feature Extraction

featureDict = containers.Map;   % Create features dictionary
fid = fopen(allFiles);  % Open file with directory of all audio files
myData = textscan(fid,'%s');    % Read the file
fclose(fid);    % Close the file
myFiles = myData{1};    % myData{1} contains a 565x1 cell that has all the audio file directories

for cnt = 1:length(myFiles) % For each of the 565 files
    [audioIn,fs] = audioread(myFiles{cnt}); % Extract sample data and sample rate
    [F0,lik] = fast_mbsc_fixedWinlen_tracking(audioIn,fs);  % Estimated pitch (F0) and lik = frame degree of voicing for EACH FRAME -> F0 & lik are 500x1 column vector
    avg_F0 = mean(F0(lik>0.45));
    [coeff] = v_melcepst(audioIn,fs,'Mtaz',MFCC_Num); % MFCC; add 'dD' for Second delta MCFF, Third delta delta MCFF
    coeff_avg = mean(coeff);
    featureDict(myFiles{cnt}) = cat(2,avg_F0,coeff_avg);
    if(mod(cnt,5)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end


%%
% Train Read, Test Read
trainList = 'train_read.txt';  
testList = 'test_read.txt';
% Train the Classifier
[trainFeatures, trainLabels, num_features] = trainClassifier(trainList, featureDict);
Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

% Test the classifier
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(1,1) = eer;


%%
% Train Read, Test Phone
trainList = 'train_read.txt';  
testList = 'test_phone.txt';
% Train the Classifier
[trainFeatures, trainLabels, num_features] = trainClassifier(trainList, featureDict);
Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

% Test the classifier
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(1,2) = eer;


%%
% Train Read, Test Mismatch
trainList = 'train_read.txt';  
testList = 'test_mismatch.txt';
% Train the Classifier
[trainFeatures, trainLabels, num_features] = trainClassifier(trainList, featureDict);
Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

% Test the classifier
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(1,3) = eer;

%%
% Train Phone, Test Read
trainList = 'train_phone.txt';
testList = 'test_read.txt';
% Train the Classifier
[trainFeatures, trainLabels, num_features] = trainClassifier(trainList, featureDict);
Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

% Test the classifier
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(2,1) = eer;


%%
% Train Phone, Test Phone
trainList = 'train_phone.txt';
testList = 'test_phone.txt';
% Train the Classifier
[trainFeatures, trainLabels, num_features] = trainClassifier(trainList, featureDict);
Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

% Test the classifier
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(2,2) = eer;

%%
% Train Phone, Test Mismatch
trainList = 'train_phone.txt';
testList = 'test_mismatch.txt';
% Train the Classifier
[trainFeatures, trainLabels, num_features] = trainClassifier(trainList, featureDict);
Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',15000,'Standardize',1);

% Test the classifier
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(2,3) = eer;

disp(EER_Matrix);
toc




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [trainFeatures, trainLabels, num_col] = trainClassifier(trainList, featureDict)
	% Train the classifier
	fid = fopen(trainList); % Opens trainList file
	myData = textscan(fid,'%s %s %f');  % Read in first file, second file, whether or not speaker is the same
	fclose(fid);    % Close file
	fileList1 = myData{1};  % First column of trainList
	fileList2 = myData{2};  % Second column of trainList
	trainLabels = myData{3};    % Third column of trainList (whether or not same speaker)
	num_row = length(trainLabels);
	num_col = length(featureDict(fileList1{1}));
	trainFeatures = zeros(num_row,num_col);
	for cnt = 1:length(trainLabels)
	    trainFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt})); % Finds difference between avg F0 of first file and second file
	end
end

function [testFeatures, testLabels] = testClassifier(testList, num_features, featureDict)
	% Test the classifier
	fid = fopen(testList);
	myData = textscan(fid,'%s %s %f');
	fclose(fid);
	fileList1 = myData{1};
	fileList2 = myData{2};
	testLabels = myData{3};
	testFeatures = zeros(length(testLabels),num_features);
	for cnt = 1:length(testLabels)
	    testFeatures(cnt,:) = -abs(featureDict(fileList1{cnt})-featureDict(fileList2{cnt}));
	end
end
