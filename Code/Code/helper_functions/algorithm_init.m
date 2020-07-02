function [Gamma_est] = algorithm_init(Gamma_est,f,N,K,M)
[A] = convert_for_comp(f);
Z = zeros(N,K,M);
for i = 1:size(A,1)
 Z(A(i,1),A(i,3),A(i,2)) = 1;
end

[n,k,m] = size(Z);
q = mean(Z,3);
q = q ./ repmat(sum(q,2),1,k);
mu = zeros(k,k,m);
for i = 1:m
    mu(:,:,i) = (Z(:,:,i))'*q;
    if(isempty(Gamma_est{i}))
        Gamma_est{i} = mu(:,:,i);
        Gamma_est{i} = Gamma_est{i}*diag(1./sum(Gamma_est{i},1));
    end
end

end