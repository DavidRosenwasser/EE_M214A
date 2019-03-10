%##############################################################
% Script to perform speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2019
% Hamza Zamani, David Rosenwasser, & Taishi Kato
%##############################################################

% Description
% This is the main code that does feature extraction, model training, and model testing in order to compute the EER for speaker verification.
% 

clear all;
clc;
%%

% Definitions
allFiles = 'allFiles.txt';
EER_Matrix = zeros(2,3);
MFCC_Num = 12; % Number of MFCCs to extract
num_neighbors = 75; % Number of neighbors used in KNN
pre_emph = [1 -0.95]; % Pre-emphasis filter parameters

tic
%% 
% Feature Extraction
% This part of the code extracts the features of interest from each audio file

featureDict = containers.Map;   % Create features dictionary
fid = fopen(allFiles);  % Open file with directory of all audio files
myData = textscan(fid,'%s');    % Read the file
fclose(fid);    % Close the file
myFiles = myData{1};    % myData{1} contains a 565x1 cell that has all the audio file directories

for cnt = 1:length(myFiles) % For each all of the files
    [audioIn,fs] = audioread(myFiles{cnt}); % Extract sample data and sample rate
	audioIn = filter(pre_emph,1,audioIn); % Apply pre-emphasis filter
    %[F0,lik] = fast_mbsc_fixedWinlen_tracking(audioIn,fs);  % Pitch, not used.
    %avg_F0 = mean(F0(lik>0.45)); % Taking the average pitch, not used.
    [coeff] = v_melcepst(audioIn,fs,'Mtaz',MFCC_Num); % Finding MFCCs; add 'dD' for Second delta MCFF, Third delta delta MCFF
    coeff_avg = mean(coeff);
    coeff_std = std(coeff);
    featureDict(myFiles{cnt}) = cat(2,coeff_avg,coeff_std); % Combine mean and std vectors to form feature vector
    if(mod(cnt,10)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end


%%
% This part of the code will train the model on read-read pairs and then test on read-read pairs
trainList = 'train_read.txt';  
testList = 'test_read.txt';
% Train the Classifier
[trainFeatures, trainLabels, num_features] = trainClassifier(trainList, featureDict);
Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',num_neighbors,'Distance','seuclidean','Standardize',1);

% Test the classifier on read-read pairs and calculate EER
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(1,1) = eer;


%%
% Using the previous model trained on read-read pairs, we will test on phone-phone pairs.
testList = 'test_phone.txt';
% Use Mdl from previous training

% Test the classifier on phone-phone pairs and calculate EER
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(1,2) = eer;


%%
% Using the previous model trained on read-read pairs, we will test on mismatched pairs
testList = 'test_mismatch.txt';
% Use Mdl from previous training

% Test the classifier on mismatched pairs and calculate EER
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(1,3) = eer;

%%
% This part of the code will train the model on phone-phone pairs and then test on read-read pairs
trainList = 'train_phone.txt';
testList = 'test_read.txt';
% Train the Classifier
[trainFeatures, trainLabels, num_features] = trainClassifier(trainList, featureDict);
Mdl = fitcknn(trainFeatures,trainLabels,'NumNeighbors',num_neighbors,'Distance','seuclidean','Standardize',1);

% Test the classifier on read-read pairs and calculate EER
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(2,1) = eer;


%%
% Using the previous model trained on phone-phone pairs, we will test on phone-phone pairs
testList = 'test_phone.txt';
% Use Mdl from previous training

% Test the classifier on phone-phone pairs and calculate EER
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(2,2) = eer;

%%
% Using the previous model trained on phone-phone pairs, we will test on mismatched pairs
testList = 'test_mismatch.txt';
% Use Mdl from previous training

% Test the classifier on mismatched pairs and calculate EER
[testFeatures, testLabels] = testClassifier(testList, num_features, featureDict);
[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
EER_Matrix(2,3) = eer;

disp(EER_Matrix);

toc



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% This function trains the classifier 
% Inputs:
%	- trainList: Text file that contains the training list of speakers to train the classifier
%	- featureDict: Feature dictionary for all of the audio files
%
% Outputs:
%	- trainFeatures: The negative absolute difference between feature vectors of the two speakers on the trainList
%	- trainLabels:	The third column of the trainList file which shows whether or not the two speakers are the same
%	- num_col:	This is the number of columns in the featureDict which singifies the number of features

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

% This function tests the classifier 
% Inputs:
%	- testList: Text file that contains the testing list of speakers to test the classifier
%	- featureDict: Feature dictionary for all of the audio files
%
% Outputs:
%	- testFeatures: The negative absolute difference between feature vectors of the two speakers on the testList
%	- testLabels:	The third column of the testList file which shows whether or not the two speakers are the same

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
