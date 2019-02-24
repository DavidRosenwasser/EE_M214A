function [net] = TrainNet(trainFeatures,trainLabels)
%% Description
% This function takes in the feature dictionary, training data, and test
% data and returns the ERR and the trained net. Currently using a
% feedforward net
% 
% Inputs:
%   trainFeatures: collected features for each training instance. this
%   variable is created by taking features from both speech files and
%   processing them whether it's subraction of the values, concatination,
%   etc. Expects a matrix where features from each test are contain in
%   column vectors. For example, if 10 features are collected from the two
%   input files and there are 50 tests, the matrix has a dimention of 10x50:
%       
%       size(trainFeatures) = [10 50]
%      
%   trainLabels: is a row vector containing the labels for each test. each
%   element of the row vector cooresponding with the test between the two
%   files in trainFeatures. So for the above trainFeatures the matching 
%   trainLabels is 1x50:
%
%       size(trainFeatures) = [10 50]
%
%   trainFeatures: collected features for each testing instance. this
%   variable follows the same sizing convention as trainFeatures
%
%   trainLabels: collected features for each testing instance. this
%   variable follows the same sizing convention as trainLabels  
%
% Outputs:    
%   - net: Trained neural net object trained with training data and tested
%           with the testing data to produce ERR

%% debug constants
verbose = 0;

%% Train net with training data
% set up intermediate variables for ease of debug
% training input matrix
x = trainFeatures;
% training label matrix
t = trainLabels;

% instantiate feed forward net with 3 layers of 20 neurons each
% tune as needed
net = feedforwardnet([10 10 10]);

% configure training/validation/testing ratio
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% try adaptive training??

% train the network and display progress
[net,tr] = train(net,x,t);
if(verbose == 1)
    view(net)
    nntraintool
    plotperform(tr) 
end


