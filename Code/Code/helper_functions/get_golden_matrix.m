function [X] = get_golden_matrix(Y,M,I)
N=length(I);

L=1;%ceil(N/M);
X= cell(L,2);
N_list = 1:N;
for k=1:L
X1=cell((N-M),1);
for i=N_list(M+1:N)
    X1{i}=[];
    for j=N_list(1:M)
        X1{i} = [X1{i} ; Y{j,i}];
    end
    X{k,1}=[X{k,1} X1{i}];
end
%X{k,1} = X{k,1}*diag(1./sum(X{k,1},1));
X{k,2}= N_list(1);
N_list=circshift(N_list,-M);
end
end