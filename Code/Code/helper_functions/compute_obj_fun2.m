function [obj_fun] = compute_obj_fun2(M_mat,mean_vec,Psi,p_vec,compute_cov)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 5
    compute_cov = 1;
end

if compute_cov
    P = diag(p_vec) - p_vec*p_vec';
else
    P = diag(p_vec);
end

M = size(M_mat,1);
obj_fun = 0;
for i=1:M
    for j=i:M
        if i~=j
            obj_fun = obj_fun + 0.5*norm(M_mat{i,j} - Psi{i}*P*Psi{j}','fro')^2;
        end
    end
    obj_fun = obj_fun + 0.5*norm(mean_vec(:,i) - Psi{i}*p_vec)^2;
end
end

