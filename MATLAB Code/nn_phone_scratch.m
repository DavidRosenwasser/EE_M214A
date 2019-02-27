%##############################################################
% Script to perform speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2019
% Hamza Zamani, David Rosenwasser, & Taishi Kato
%##############################################################

 
%%
clear all;
clc;
%% config
vert_cat = 0;
%%
% Define lists
allFiles = 'allFiles.txt';
trainList_read = 'train_read.txt';  
testList_read = 'test_read.txt';
trainList_phone = 'train_phone.txt';  
testList_phone = 'test_phone.txt';
trainList_mismatch = 'train_mismatch.txt';  
testList_mismatch = 'test_mismatch.txt';
%% 
% Feature Extraction of all files
FeatureDict = containers.Map;       % Create features dictionary
fid = fopen(allFiles);              % Open file with directory of all audio files
myData = textscan(fid,'%s');        % Read the file
fclose(fid);                        % Close the file
myFiles = myData{1};                % myData{1} contains a 565x1 cell that has all the audio file directories

for cnt = 1:length(myFiles)         % For each of the 565 files
    [audioIn,fs] = audioread(myFiles{cnt});                 % Extract sample data and sample rate
    %[F0,lik] = fast_mbsc_fixedWinlen_tracking(audioIn,fs);  % Estimated pitch (F0) and lik = frame degree of voicing for EACH FRAME -> F0 & lik are 500x1 column vector
    %avg_F0 = mean(F0(lik>0.45));
    [coeff] = v_melcepst(audioIn,fs,'M0tazdD');
    coeff_avg = mean(coeff);
    m = mean(coeff_avg);
    s = std(coeff_avg);       
    % intermediate variable prior to adding to the dictionary 
    %entry = horzcat(avg_F0, m, s, coeff_avg);
    entry = horzcat(m, s, coeff_avg);
    FeatureDict(myFiles{cnt}) = entry';
    if(mod(cnt,5)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end



%%
disp('Collect samples training samples associated with testing and reshape into');
% training data format
fid = fopen(trainList_phone);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
trainLabels = myData{3};

%
disp('format the label (or target) matrix');
% format the label (or target) matrix 
% check orientation of matrix - MUST BE A ROW VECTOR. Therefore check the
% dimensions and transpose if necessary
labelDim = size(trainLabels);
% if label data is a column vector i.e. rows > 1
if(labelDim(1) ~= 1)     
    trainLabels = trainLabels';
end
    
% format training input matrix using extracted feature dictionary

% # of rows for training matrix. numRows = 2*(# of extracted features).      
% Multiply by two since our training data consists of 2 speech samples
% to be compared
if (vert_cat == 1)
    numRows = length(FeatureDict(fileList1{cnt}))*2;
else
    numRows = length(FeatureDict(fileList1{cnt}));
end

% # of col for training matrix. numCol = # of labels (test cases).
% MUST match number of test cases.
numCol = length(trainLabels);

% init zero matrix for speed
trainFeatures = zeros(numRows,numCol);

for cnt = 1:length(trainLabels)
    % use the filename to look up value in dictionary for both files and
    % concatinate into column vector
    if(vert_cat == 1)
        trainFeatures(:,cnt) = vertcat(FeatureDict(fileList1{cnt}), FeatureDict(fileList2{cnt}));
    else
        trainFeatures(:,cnt) = -abs(FeatureDict(fileList1{cnt}) - FeatureDict(fileList2{cnt}));
    end
end

% Neural net instantiation and training
disp('Neural net instantiation and training');
% since the output is binary the neural net is configure for a single
% output bin.
setdemorandstream(391418381)
% setting up intermediate variables for ease of debug training input matrix
x = trainFeatures;
% training label matrix
t = trainLabels;
% set seed for initial untrained weights
%setdemorandstream(391418381)
net = feedforwardnet([30 40 50]);
%view(net)
% train the network
[net,tr] = train(net,x,t);
nntraintool
%% PHONE-READ
% now extract data from test_read.txt and input to network
disp('extract data from test_read.txt');
% extract features
% Test the classifier
fid = fopen(testList_phone);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1_test = myData{1};
fileList2_test = myData{2};
testLabels = myData{3};

% format the label (or target) matrix 
% check orientation of matrix - MUST BE A ROW VECTOR. Therefore check the
% dimensions and transpose if necessary
labelDim = size(testLabels);
% if label data is a column vector i.e. rows > 1
if(labelDim(1) > 1)     
    testLabels = testLabels';
end
    
% format training input matrix using extracted feature dictionary

% # of rows for training matrix. numRows = 2*(# of extracted features).      
% Multiply by two since our training data consists of 2 speech samples
% to be compared
if (vert_cat == 1)
    numRows = length(FeatureDict(fileList1{1}))*2;
else
    numRows = length(FeatureDict(fileList1{1}));
end

% # of col for training matrix. numCol = # of labels (test cases).
% MUST match number of test cases.
numCol = length(testLabels);

% init zero matrix for speed
testFeatures = zeros(numRows,numCol);

for cnt = 1:length(testLabels)
    % use the filename to look up value in dictionary for both files and
    % concatinate into column vector
    if(vert_cat == 1)
        testFeatures(:,cnt) = vertcat(FeatureDict(fileList1_test{cnt}), FeatureDict(fileList2_test{cnt}));
    else
        testFeatures(:,cnt) = -abs(FeatureDict(fileList1_test{cnt}) - FeatureDict(fileList2_test{cnt}));
    end
end

% now attempt to feed neural net with test data
disp('feed neural net test data');
% run neural net
testOutput = net(testFeatures);
%testScores = (testOutput(:,2)./(testOutput(:,1)+1e-15));
[eer,~] = compute_eer(testOutput, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
roundedOutputs = round(testOutput);
errors = 0;
% check neural net outputs vs acutal labels
for i=1:length(testOutput)
    if(testLabels(i) ~= roundedOutputs(i))
        errors = errors + 1;
    end
end

%% PHONE-PHONE
disp('extract data from test_phone.txt');
% extract features
% Test the classifier
fid = fopen(testList_phone);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1_test = myData{1};
fileList2_test = myData{2};
testLabels = myData{3};

% format the label (or target) matrix 
% check orientation of matrix - MUST BE A ROW VECTOR. Therefore check the
% dimensions and transpose if necessary
labelDim = size(testLabels);
% if label data is a column vector i.e. rows > 1
if(labelDim(1) > 1)     
    testLabels = testLabels';
end
    
% format training input matrix using extracted feature dictionary

% # of rows for training matrix. numRows = 2*(# of extracted features).      
% Multiply by two since our training data consists of 2 speech samples
% to be compared
if (vert_cat == 1)
    numRows = length(FeatureDict(fileList1{1}))*2;
else
    numRows = length(FeatureDict(fileList1{1}));
end

% # of col for training matrix. numCol = # of labels (test cases).
% MUST match number of test cases.
numCol = length(testLabels);

% init zero matrix for speed
testFeatures = zeros(numRows,numCol);

for cnt = 1:length(testLabels)
    % use the filename to look up value in dictionary for both files and
    % concatinate into column vector
    if(vert_cat == 1)
        testFeatures(:,cnt) = vertcat(FeatureDict(fileList1_test{cnt}), FeatureDict(fileList2_test{cnt}));
    else
        testFeatures(:,cnt) = -abs(FeatureDict(fileList1_test{cnt}) - FeatureDict(fileList2_test{cnt}));
    end
end


% run neural net
testOutput = net(testFeatures);
%testScores = (testOutput(:,2)./(testOutput(:,1)+1e-15));
[eer,~] = compute_eer(testOutput, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
roundedOutputs = round(testOutput);
errors = 0;
% check neural net outputs vs acutal labels
for i=1:length(testOutput)
    if(testLabels(i) ~= roundedOutputs(i))
        errors = errors + 1;
    end
end

%% PHONE-MISMATCH
disp('extract data from test_mismatch.txt');
% extract features
% Test the classifier
fid = fopen(testList_mismatch);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1_test = myData{1};
fileList2_test = myData{2};
testLabels = myData{3};

% format the label (or target) matrix 
% check orientation of matrix - MUST BE A ROW VECTOR. Therefore check the
% dimensions and transpose if necessary
labelDim = size(testLabels);
% if label data is a column vector i.e. rows > 1
if(labelDim(1) > 1)     
    testLabels = testLabels';
end
    
% format training input matrix using extracted feature dictionary

% # of rows for training matrix. numRows = 2*(# of extracted features).      
% Multiply by two since our training data consists of 2 speech samples
% to be compared
if (vert_cat == 1)
    numRows = length(FeatureDict(fileList1{1}))*2;
else
    numRows = length(FeatureDict(fileList1{1}));
end

% # of col for training matrix. numCol = # of labels (test cases).
% MUST match number of test cases.
numCol = length(testLabels);

% init zero matrix for speed
testFeatures = zeros(numRows,numCol);

for cnt = 1:length(testLabels)
    % use the filename to look up value in dictionary for both files and
    % concatinate into column vector
    if(vert_cat == 1)
        testFeatures(:,cnt) = vertcat(FeatureDict(fileList1_test{cnt}), FeatureDict(fileList2_test{cnt}));
    else
        testFeatures(:,cnt) = -abs(FeatureDict(fileList1_test{cnt}) - FeatureDict(fileList2_test{cnt}));
    end
end


% run neural net
testOutput = net(testFeatures);
%testScores = (testOutput(:,2)./(testOutput(:,1)+1e-15));
[eer,~] = compute_eer(testOutput, testLabels);
disp(['The EER is ',num2str(eer),'%.']);
roundedOutputs = round(testOutput);
errors = 0;
% check neural net outputs vs acutal labels
for i=1:length(testOutput)
    if(testLabels(i) ~= roundedOutputs(i))
        errors = errors + 1;
    end
end

