function [error_KOS] = KOS(A,y,valid_index,n,k,m)
%===================== Sewoung Oh ================

t = zeros(n,k-1);
for l = 1:k-1
    U = zeros(n,m);
    for i = 1:size(A,1)
        U(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
    end
    
    B = U - ones(n,1)*(ones(1,n)*U)/n;
    [U S V] = svd(B);
    u = U(:,1);
    v = V(:,1);
    u = u / norm(u);
    v = v / norm(v);
    pos_index = find(v>=0);
    if sum(v(pos_index).^2) >= 1/2
        t(:,l) = sign(u);
    else
        t(:,l) = -sign(u);
    end
end

J = ones(n,1)*k;
for j = 1:n
    for l = 1:k-1
        if t(j,l) == -1
            J(j) = l;
            break;
        end
    end
end
error_KOS = mean(y(valid_index) ~= (J(valid_index)))
end