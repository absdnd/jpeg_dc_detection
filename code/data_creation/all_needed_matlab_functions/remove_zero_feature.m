function [training_new,testing_new,indices] = remove_zero_feature(training,testing)
% total = vertcat(training,testing);
indexes = [];
training_new = training;
testing_new = testing;
%last column is label
for i = 1:size(training,2)-1
    feat = training(:,i);
    %     val = feat == 0;
    %     %checl if the complete feature vectore is 0
    %     if (sum(val) == size(training,1))
    %         %if complete column is zero then remove it
    %         indexes = horzcat(indexes,i);
    %     end
    val = std(feat);
    %checl if the complete feature vectore is almost same (zero std)
    if val == 0
        %if complete column is zero then remove it
        indexes = horzcat(indexes,i);
    end
end
%Removing those columns which are comletely 0
indexes = flip(indexes);
for i = 1:length(indexes)
    training_new(:,indexes(i)) = [];
    testing_new(:,indexes(i)) = [];
end

setA = 1:size(training,2);
setB = indexes;

indices = setdiff(setA,setB);
end
