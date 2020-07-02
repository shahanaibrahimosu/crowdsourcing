function [EM_err,mu] = run_EM_MV(Z,y,valid_index,Nround)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[n,k,m] = size(Z);
q = mean(Z,3);
q = q ./ repmat(sum(q,2),1,k);
mu = zeros(k,k,m);
% EM update
for iter = 1:Nround
    for i = 1:m
        mu(:,:,i) = (Z(:,:,i))'*q;
        
        mu(:,:,i) = AggregateCFG(mu(:,:,i),0);
        tmp_vec = sum(mu(:,:,i));
        indx = find(tmp_vec > 0);
        mu(:,indx,i) = bsxfun(@rdivide,mu(:,indx,i),tmp_vec(indx));
    end
    
    q = zeros(n,k);
    for i=1:m
        tmp = mu(:,:,i);
        tmp(find(tmp ==0)) = eps;
        q = q + Z(:,:,i)*log(tmp);
    end
    
    q = exp(q);
    q = bsxfun(@rdivide,q,sum(q,2));
end
[I,J] = max(q');
error_EM_predict = mean(y(valid_index) ~= (J(valid_index))');
EM_err = error_EM_predict(end);

end

