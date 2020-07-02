function [rec_error,H] = computeKLError(X,W,H,N,K)  
    H=H';
    H(H<0)=0;  
    rec_error=zeros(1,N-1);
    for i=1:N-1
        R=X(:,K*(i-1)+1:K*i);
        S=W*H(:,K*(i-1)+1:K*i);
        div = sum(S,'all');
        D=(1/div)*eye(size(S,1));
        S=D*S;
        H(:,K*(i-1)+1:K*i)=D*H(:,K*(i-1)+1:K*i);
        R=reshape(R,[size(R,1)*size(R,2),1]);
        S=reshape(S,[size(S,1)*size(S,2),1]);
        rec_error(i)=dot(R,log(R)-log(S));
    end
    H=H';
end