function  [Recall,Precision,F1,First] = Analysis_kest(skeleton,kest)
% This function evaluates the accuracy of causal order(kest) compared with the skeleton.  ----Yana
% Input:
%      skeleton: row points to column. Skeleton(i,j) means xi -> xj.


n = length(kest); %the number of variables
g = zeros(n,n); %estimated graph according to kest(causal order)
for i = 1:n
    for j = i+1:n
        g(kest(i),kest(j)) = 1;
    end
end

trueed = 0;
for i = 1:n
    for j = 1:n
        if g(i,j) ==1 && skeleton(i,j) ==1
            trueed = trueed+1;
        end
    end
end

alled_true_graph = sum(sum(skeleton));%all edges in true graph
alled_pre_graph = sum(sum(g));
Precision = trueed/alled_pre_graph;
Recall = trueed/alled_true_graph;
F1 = 2*Precision*Recall/(Precision+Recall);


if all(skeleton(:,kest(1))==0)
    First = 1;
else
    First =0;
end
end