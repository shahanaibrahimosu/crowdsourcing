function [EM_err,mu] = run_EM(mu,Z,y,valid_index,Nround)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[n,k,m] = size(Z);
% EM update
for iter = 1:Nround
    q = zeros(n,k);
    for i=1:m
        tmp = mu(:,:,i);
        tmp(find(tmp ==0)) = eps;
        tmp(find(isnan(tmp))) = eps;
        q = q + Z(:,:,i)*log(tmp);
        %q = q + Z(:,:,i)*log(mu(:,:,i));
    end
    q = exp(q);
    q = bsxfun(@rdivide,q,sum(q,2));

    for i = 1:m
        mu(:,:,i) = (Z(:,:,i))'*q;
        
        mu(:,:,i) = AggregateCFG(mu(:,:,i),0);
        tmp_vec = sum(mu(:,:,i));
        indx = find(tmp_vec > 0);
        mu(:,indx,i) = bsxfun(@rdivide,mu(:,indx,i),tmp_vec(indx));
    end
end
[I,J] = max(q');
error_EM_predict = mean(y(valid_index) ~= (J(valid_index))');
EM_err = error_EM_predict(end);

end

