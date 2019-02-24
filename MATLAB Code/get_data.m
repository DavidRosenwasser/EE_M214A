function [fileList1,fileList2, labels] = GetData(filename)
% Formats files into matricies with 
	% Train the classifier
	fid = fopen(filename); % Opens trainList file
	myData = textscan(fid,'%s %s %f');  % Read in first file, second file, whether or not speaker is the same
	fclose(fid);    % Close file
	fileList1 = myData{1};  % First column of trainList
	fileList2 = myData{2};  % Second column of trainList
	labels = myData{3};    % Third column of trainList (whether or not same speaker)
end

