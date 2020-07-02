function [A_mat,A_ind,A_tens] = convert_to_adjacency(f)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = size(f,2);
M = size(f,1);
A_ind = cell(M,1);
A_mat = sparse(N*N,M);
A_tens = zeros(N,N,M);
zz = unique(f);
if any(ismember(zz,0))
    K = numel(zz) - 1; 
else
    K = numel(zz);
end

for m=1:M
    %Tmp = squareform(pdist(f(m,:)'));
	Tmp = zeros(N,N);
	for k=1:K
		y = zeros(N,1);
		y(f(m,:) == k) = 1;
		Tmp = Tmp + y*y';
	end
    indx = find(Tmp == 0);
    %A_ind{m} = sparse(N,N);
    %A_ind{m} = zeros(N);
    %A_ind{m}(indx) = 1;
	%A_ind{m} = A_ind{m} - eye(N);
    %indx2 = find(A_ind{m});
    A_mat(indx,m) = 1;
    A_ind{m} = sparse(A_ind{m});
    A_tens(:,:,m) = A_ind{m};
end
A_mat = sparse(A_mat);
%A_tens = sparse(A_tens);

end

