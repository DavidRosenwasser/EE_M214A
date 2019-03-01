function [EER] = TestNet(net, testFeatures, testLabels)
%% Description
% This function takes in a trained neural net and run test data through and
% returns the EER metric

%% Run test data through trained net
testOutput = net(testFeatures);
[EER,~] = compute_eer(testOutput, testLabels);