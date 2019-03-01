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

