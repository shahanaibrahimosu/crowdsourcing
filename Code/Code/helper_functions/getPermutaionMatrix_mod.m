function [Pm,A] = getPermutaionMatrix_mod(A,list_g)
K = size(A{1},2);
marg=num2cell(perms(1:K),2);
d_sum = zeros(1,size(marg,1));
max_id = zeros(1,length(list_g));
for j=1:length(list_g)
    for i=1:size(marg,1)
        I_m= eye(K);
        Pm = I_m(:,marg{i});
        d_sum(i)= trace(A{list_g(j)}*Pm);
    end
    [~,max_id(j)] = max(d_sum);
end
Pm =  I_m(:,marg{mode(max_id)});
%Pm =[1 0;0 1];

for n=list_g   
    A{n} = A{n}*Pm;
end
end