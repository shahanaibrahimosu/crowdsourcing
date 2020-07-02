function [Gamma_rot,p_vec_rot, perm_flag,perm] = rotate_conf_mat(Gamma,p_vec)
%ROTATE_CONF_MAT Function that checks whether confusion matrices need to be
%aligned.
%   Panagiotis Traganitis. traga003@umn.edu

M = length(Gamma); 
K = size(Gamma{1},1);

perm_flag = 0;
Gamma_rot = cell(M,1);
maxindx = zeros(M,K);
for i =1:M
   [~,maxindx(i,:)] = max(Gamma{i}); 
end

[temp,frq] = mode(maxindx);

temptemp = zeros(K,1);
rst = setdiff(1:K,temptemp);
%check for duplicate elements
while ~isempty(rst)
   for i=1:K
      indxx = find(temp == i);
      if numel(indxx) >= 1
          [~,ii] = max(frq(indxx));
          temptemp(indxx(ii)) = i;
      else
          temptemp(indxx) = i;
      end
   end
   rst = setdiff(1:K,temptemp);
   uindx = temptemp == 0;
   temptemp(uindx) = rst;
end

perm = 1:K;
perm(temptemp) = 1:K;
if ~isequal(perm,1:K)
    perm_flag = 1;
end


for i =1:M
   Gamma_rot{i} = Gamma{i}(:,perm); 
end
p_vec_rot = p_vec(perm);
end

