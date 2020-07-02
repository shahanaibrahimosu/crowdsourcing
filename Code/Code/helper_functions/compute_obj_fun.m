function [obj_fun] = compute_obj_fun(M_mat,Psi,p_vec,compute_cov)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if compute_cov 
    P = diag(p_vec) + p_vec'*p_vec;
else
    P = diag(p_vec);
end

M = size(M_mat,1);
obj_fun = 0;
for i=1:M
    for j=1:M
        if i~=j
            obj_fun = obj_fun + 0.5*norm(M_mat{i,j} - Psi{i}*P*Psi{j}','fro');
        end
    end
end
end

