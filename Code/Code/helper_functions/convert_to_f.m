function [F,f,valid_index,y] = convert_to_f(A,B)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = max(A(:,1));
M = max(A(:,2));
K = max(B(:,2));

f = zeros(M,N);
F = cell(M,1);

y = zeros(N,1); %ground-truth labels
for i = 1:size(B,1)
    y(B(i,1)) = B(i,2);
end
valid_index = find(y > 0);

for i=1:M
    indx = find(A(:,2) == i);
    indx2 = A(indx,1);
    f(i,indx2) = A(indx,3); 
end

for i=1:M 
    indx = find(f(i,:) > 0);
    F{i} = sparse(f(i,indx),indx,1,K,N);
end

end

