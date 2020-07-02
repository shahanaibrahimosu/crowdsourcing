function [A_est_G,p_vec_hat,list_good_annot] = EstConfMat_SPA(M_mat,K)

N = size(M_mat,1);
I= K*ones(1,N);

[X_cell,list_f] =get_golden_matrix_full_rank(M_mat,I);
size_l=zeros(N,1);
for i=1:N
    size_l(i,1) = size(X_cell{i},2);
end
[~,index_spa] = max(size_l);
A_est_G = cell(N,1);


k=index_spa;
X=X_cell{k};
if(size(X,2)>7*K)
    %%%%%%SPA Algorithm
    X = X*diag(1./sum(X,1));
    X(isnan(X))=0;
    [l,~,~]=FastSepNMF(X,K,0);
    W_est= X(:,l);
    %SPArectify(l,X)
    A_est_G{k} = W_est;
    A_est_G{k} = A_est_G{k}*diag(1./sum(A_est_G{k},1));
end


for k=[1:index_spa-1 index_spa+1:N]
    X=X_cell{k};
    if(size(X,2)>7*K)
        %%%%%%SPA Algorithm
        X = X*diag(1./sum(X,1));
        X(isnan(X))=0;
        [l,~,~]=FastSepNMF(X,K,0);
        W_est= X(:,l);
        A_est_G{k} = W_est;
        A_est_G{k} = A_est_G{k}*diag(1./sum(A_est_G{k},1));
    end
end


p_vec_hat = ones(K,1)/K; 


list_good_annot = [];
list_bad_annot = [];
for i=1:N
    if(~isempty(A_est_G{i}) && rank(A_est_G{i})==K)
        list_good_annot=[list_good_annot i];
    else
        list_bad_annot=[list_bad_annot i];
    end
end







for i=1:length(A_est_G)
    A_est_G{i};
    G =A_est_G{i};
    G = max( G, 10^-6 );
    t=sum(G,1);
    G = G*diag(1./t);
    A_est_G{i}=G;
    A_est_G{i};
end



end