%##############################################################
% Script to perform speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2019
% Hamza Zamani, David Rosenwasser, & Taishi Kato
%##############################################################

clear all;
clc;
%%
% Define lists
allFiles = 'allFiles.txt';
trainList = 'train_read.txt';  
testList = 'test_read.txt';

%% 
% Feature Extraction

featureDict = containers.Map;   % Create features dictionary
fid = fopen(allFiles);  % Open file with directory of all audio files
myData = textscan(fid,'%s');    % Read the file
fclose(fid);    % Close the file
myFiles = myData{1};    % myData{1} contains a 565x1 cell that has all the audio file directories

%{
for cnt = 1:length(myFiles) % For each of the 565 files
    [audioIn,fs] = audioread(myFiles{cnt}); % Extract sample data and sample rate
    [F0,lik] = fast_mbsc_fixedWinlen_tracking(audioIn,fs);  % Estimated pitch (F0) and lik = frame degree of voicing for EACH FRAME -> F0 & lik are 500x1 column vector
    featureDict(myFiles{cnt}) = mean(F0(lik>0.45)); % Finds the F0s that are most likely from voiced regions and associates with file name and then takes average
    if(mod(cnt,5)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end
%}

[coeffs,delta,deltaDelta,loc] = mfcc(audioIn,fs,'NumCoeffs',20);