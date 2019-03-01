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