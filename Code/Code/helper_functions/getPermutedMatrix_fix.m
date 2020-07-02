function [Pm,A] = getPermutedMatrix_fix(A,list_g,size_l,Y,l)

A_p = A;
[~,index_array] = sort(size_l,'descend');
j=index_array(1);
K=size(A{j},2);
take_flag =zeros(K,1);
fill_flag = zeros(K,1);
A_p{j}=zeros(size(A{j}));
M=size(A,1);
flag_rotate = zeros(1,M);
for i=1:K
   [~,ind] = max(A{j}(i,:));
   if(fill_flag(i)==0)
    fill_flag(i)=1;
    A_p{j}(:,i) = A{j}(:,ind);
    take_flag(ind)=1;
   end
end
flag_rotate(j)=1;

for i=1:K
    if(fill_flag(i)==0)
        while(1)
         sel=randsample(K,1);
         if(take_flag(sel)==0)
             break;
         end
        end
        A_p{j}(:,i) = A{j}(:,sel);
    end       
end

for i=list_g 
    if(flag_rotate(i)==0)
        for index=index_array'
            if(i~=index && rank(Y{i,index})==K)
                A_m = Y{i,index}*pinv(diag(l)*A_p{index}'); 
                A_m = A_m*diag(1./sum(A_m,1));
                [~,Pm] = perm2match(A{i},A_m);
                A_p{i}=A{i}*Pm;
                flag_rotate(i)=1;
                break;
                A_p{i}
                A{i}
            end
        end
    end
end

        
A=A_p;
end
