function [X_comb,list_f] = get_golden_matrix_full_rank(Y,I)
N=length(I);
K=length(Y{1,2});
X_comb = cell(N,1);
list_f = cell(N,1);
for i=1:N
    X=[];
    list=[];
    for j=[1:i-1 i+1:N]
        if(rank(Y{i,j})== K)
            X = [X Y{i,j}];
            list =[list j];
        end
    end
    X_comb{i}=X;
    list_f{i} = list;
end
end