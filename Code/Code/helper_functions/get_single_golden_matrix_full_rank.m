function [X_comb,list_f] = get_single_golden_matrix_full_rank(Y,I)
N=length(I);
K=length(Y{1,2});
X_comb = cell(N,1);
list_f = cell(N,1);
index=0;
for i=1:N
    list=[];
    for j=[1:i-1 i+1:N]
        if(rank(Y{i,j})== K)
            list =[list j];
        end
    end
    list_f{i} = list;
    if(length(list)==N-1)
        index=i;
        break;
    end
end

X=[];
for j=[1:index-1 index+1:N]
    if(rank(Y{index,j})== K)
        X = [X Y{i,j}];
        list =[list j];
    end
end

end