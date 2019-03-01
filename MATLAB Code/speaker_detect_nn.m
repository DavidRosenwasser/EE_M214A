%##############################################################
% Script to perform speaker verficiation (NN)
% ECE214A: Digital Speech Processing, Winter 2019
% Hamza Zamani, David Rosenwasser, & Taishi Kato
%##############################################################

%% neural net implementation
clear;
clc;
%%
% Definitions
allFiles = 'allFiles.txt';
EER_Matrix = zeros(2,3);
MFCC_Num = 20;

%% Feature Extraction of all files
FeatureDict = containers.Map;       % Create features dictionary
fid = fopen(allFiles);              % Open file with directory of all audio files
myData = textscan(fid,'%s');        % Read the file
fclose(fid);                        % Close the file
myFiles = myData{1};                % myData{1} contains a 565x1 cell that has all the audio file directories

for cnt = 1:length(myFiles)         % For each of the 565 files
    [audioIn,fs] = audioread(myFiles{cnt});                 % Extract sample data and sample rate
    [F0,lik] = fast_mbsc_fixedWinlen_tracking(audioIn,fs);  % Estimated pitch (F0) and lik = frame degree of voicing for EACH FRAME -> F0 & lik are 500x1 column vector
    avg_F0 = mean(F0(lik>0.45));
    [coeff] = v_melcepst(audioIn,fs,'M0tazdD', MFCC_Num);
    coeff_avg = mean(coeff);      
    coeff_std = std(coeff);
    coeff_avg_std = std(coeff_avg);
    % concatinate average F0 and averaged coeff
    % add features as desired here prior to adding to dictionary
    % note: adding dictionary values as col vector
    %FeatureDict(myFiles{cnt}) = horzcat(avg_F0, coeff_avg)';
    FeatureDict(myFiles{cnt}) = horzcat(coeff_avg, coeff_std, coeff_avg_std)';
    if(mod(cnt,5)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end

%% Extract all data for use later
% Read training and test  
[trainReadList1,trainReadList2, trainReadLabels] = get_data('train_read.txt');
[testReadList1,testReadList2, testReadLabels] = get_data('test_read.txt');
% Phone training and test  
[trainPhoneList1,trainPhoneList2, trainPhoneLabels] = get_data('train_phone.txt');
[testPhoneList1,testPhoneList2, testPhoneLabels] = get_data('test_phone.txt');
% Mismatch training and test
[testMismatchList1,testMismatchList2, testMismatchLabels] = get_data('test_mismatch.txt');

%% Format training data
% init for speed
% num features in coloumn vector
%row = length(FeatureDict(myFiles{cnt}));
% init - commented out to make it more flexible to update how we combine
% feature sets
%trainReadFeatures = zeros(row, length(trainReadLabels));
%trainPhoneFeatures = zeros(row, length(trainPhoneLabels));

%transpose labels for nn input (keep origninal label for debug)
trainReadLabelsRow = trainReadLabels';
trainPhoneLabelsRow = trainPhoneLabels';

% format training data for read
for i =1:length(trainReadLabels) 
    trainReadFeatures(:,i) = -abs(FeatureDict(trainReadList1{cnt})-FeatureDict(trainReadList2{cnt}));
    %trainReadFeatures(:,i) = vertcat(FeatureDict(trainReadList1{cnt}),FeatureDict(trainReadList2{cnt}));
end
for i =1:length(trainPhoneLabels)
    trainPhoneFeatures(:,i) = -abs(FeatureDict(trainPhoneList1{cnt})-FeatureDict(trainPhoneList2{cnt}));
    %trainPhoneFeatures(:,i) = vertcat(FeatureDict(trainPhoneList1{cnt}),FeatureDict(trainPhoneList2{cnt}));
end

%% Extract and format test matricies
% init - commented out to make it more flexible to update how we combine
% feature sets
%testReadFeatures = zeros(row, length(testReadLabels));
%testPhoneFeatures = zeros(row, length(testPhoneLabels));
%testMismatchFeatures = zeros(row, length(testMismatchLabels));

%transpose labels for nn input (keep origninal label for debug)
testReadLabelsRow = testReadLabels';
testPhoneLabelsRow = testPhoneLabels';
testMismatchLabelsRow = testMismatchLabels';
% Read 
for i =1:length(testReadLabels)
    % change 
    %testReadFeatures(:,i) = vertcat(FeatureDict(testReadList1{cnt}),FeatureDict(testReadList2{cnt}));
    testReadFeatures(:,i) = -abs(FeatureDict(testReadList1{cnt})-FeatureDict(testReadList2{cnt}));
end
% Phone
for i =1:length(testPhoneLabels)
    % change 
    %testPhoneFeatures(:,i) = vertcat(FeatureDict(testPhoneList1{cnt}),FeatureDict(testPhoneList2{cnt}));
    testPhoneFeatures(:,i) = -abs(FeatureDict(testPhoneList1{cnt})-FeatureDict(testPhoneList2{cnt}));
end
% Mismatch
for i =1:length(testMismatchLabels)
    % change 
    %testMismatchFeatures(:,i) = vertcat(FeatureDict(testMismatchList1{cnt}),FeatureDict(testMismatchList2{cnt}));
    testMismatchFeatures(:,i) = -abs(FeatureDict(testMismatchList1{cnt})-FeatureDict(testMismatchList2{cnt}));
end

%% create matricies for combined training and testing of all given data
% train data
trainAllLabels = vertcat(trainReadLabels, trainPhoneLabels)';
trainingAllFeatures = [trainReadFeatures trainPhoneFeatures];
% test data
testAllLabels = vertcat(testReadLabels, testPhoneLabels, testMismatchLabels)';
testAllFeatures = [testReadFeatures testPhoneFeatures testMismatchFeatures];

%% Test the net for each case
% train a net based on read data
[net_read] = TrainNet(trainReadFeatures,trainReadLabelsRow);
[net_phone] = TrainNet(trainPhoneFeatures,trainPhoneLabelsRow);
% Train Read, Test Read
EER = TestNet(net_read, testReadFeatures, testReadLabelsRow);
EER_Matrix(1,1) = EER;
disp(['The READ-READ EER is ',num2str(EER),'%.']);

% Train Read, Test Phone
EER = TestNet(net_read, testPhoneFeatures, testPhoneLabelsRow);
EER_Matrix(1,2) = EER;
disp(['The READ-PHONE EER is ',num2str(EER),'%.']);

% Train Read, Test Mismatch
EER = TestNet(net_read, testMismatchFeatures, testMismatchLabelsRow);
EER_Matrix(1,3) = EER;
disp(['The READ-MISMATCH EER is ',num2str(EER),'%.']);

% Train Phone, Test Read
EER = TestNet(net_phone, testReadFeatures, testReadLabelsRow);
EER_Matrix(2,1) = EER;
disp(['The PHONE-READ EER is ',num2str(EER),'%.']);

% Train Phone, Test Phone
EER = TestNet(net_phone, testPhoneFeatures, testPhoneLabelsRow);
EER_Matrix(2,2) = EER;
disp(['The PHONE-PHONE EER is ',num2str(EER),'%.']);

% Train Phone, Test Mismatch
EER = TestNet(net_phone, testReadFeatures, testReadLabelsRow);
EER_Matrix(2,3) = EER;
disp(['The PHONE-MISMATCH EER is ',num2str(EER),'%.']);

% Present results
disp(EER_Matrix);
%% Train neural net with all training data
% train a net with both read and phone training sets
net_comb = TrainNet(trainingAllFeatures,trainAllLabels);
% test the trained net with all labelled test vectors given (read, phone,
% mismatch)
EER_comb = TestNet(net_comb, testAllFeatures, testAllLabels);
disp(['The Total EER is ',num2str(EER_comb),'%.']);
