function [C] = khatrirao_func(A,B) %function to compute Khatri-Rao product A(.)B 
    [I,F] = size(A); J = size(B,1); %find out sizes of matrices
    C = zeros(I*J,F); %ouput matrix
    for f=1:F
       C(:,f) = reshape(B(:,f)*A(:,f)',[],1); %compute outer product of b_f and a_f and vectorize it
    end

%     [I,F] = size(A);
%     J = size(B,1);
%     C = zeros(I*J,F); %ouput matrix
%     for i=1:I
%         Ctmp = bsxfun(@times,B,A(i,:)); %Multiply each column of B with the corresponding entry from row i of A
%         indx = (i-1)*J + 1;
%         C(indx:indx+J-1,:) = Ctmp;
%     end
end