function [EER,net] = ValidateNet(trainFeatures,trainLabels, testFeatures, testLabels)
% This function trains a nn with the given trainFeatures and testFeatures
% and computes the EER when tested with the given testFeatures and testLabels
%
% Custom functions used: 
%   - TrainNet()
%   - TestNet()

%% Neural net training 
[net] = TrainNet(trainFeatures,trainLabels);

%% Neural net testing
[EER] = TestNet(net, testFeatures, testLabels);

end

