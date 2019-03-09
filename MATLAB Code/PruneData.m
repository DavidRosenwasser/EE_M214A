function [pruned_fileList1, pruned_fileList2, pruned_labels] = PruneData(training_file)

% extract data
[fileList1,fileList2, labels] = get_data(training_file);

positive_cases = find(labels);
num_positive_cases = size(positive_cases); 
training_set_size = 2*(num_positive_cases(1));

pruned_fileList1 = cell(training_set_size, 1);
pruned_fileList2 = cell(training_set_size, 1);
pruned_labels    = zeros(training_set_size, 1);
% populate the 
for i = 1:size(positive_cases)
    idx = positive_cases(i);
    pruned_fileList1{i} = fileList1{idx};
    pruned_fileList2{i} = fileList2{idx};
    pruned_labels(i)    = labels(idx);
end

for i = size(positive_cases)+1:training_set_size
   % generate random number to use as index selection
   index = randi(training_set_size);
   % append the values to the training set
   pruned_fileList1{i} = fileList1{index};
   pruned_fileList2{i} = fileList2{index};
   pruned_labels(i)    = labels(index);
end

end

