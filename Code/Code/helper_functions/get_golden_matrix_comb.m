function [X_comb] = get_golden_matrix_comb(Y,I)
N=length(I);
X_comb = cell(N,1);
for i=1:N
    X=[];
    for j=[1:i-1 i+1:N]
%         if(j<i)
        X = [X Y{i,j}];
%         else
%             X = [X Y{i,j}];
%         end
    end
    X_comb{i}=X;
end
end