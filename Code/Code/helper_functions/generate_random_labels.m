function [y,Gamma,F,f] = generate_random_labels(N,M,L,p_vec)
%GENERATE_RANDOM_LABELS Function to generate synthetic ground-truth labels,
%annotator labels, and annotator confusion matrices.
%   Input: N - number of ground-truth labels, i.e. number of data.
%          M - number of annotators
%          L - number of classes.
%      p_vec - Lx1 vector of prior probabilities, i.e. p_vec(i) = Pr(Y==i)
%  Output: y - Nx1 vector containing ground-truth labels.
%        Gamma - Mx1 cell containing M LxL confusion matrices.
%          F - Mx1 cell containing M LxN matrices, with annotator labels,
%          in vector format.
%          f - MxN matrix containing annotator labels in scalar format.
%          Each row corresponds to one annotator.
% Panagiotis Traganitis. traga003@umn.edu

if nargin < 4
    p_vec = ones(L,1)./L;
end

y = randsample(L,N,true,p_vec); %generate ground truth labels.
Gamma = generate_confusion_mat(M,L); %generate confusion matrices.

f = zeros(M,N); %annotator labels
for i=1:L
   idx = find(y == i); 
   for j=1:M
      n = numel(idx);
      tmp = randsample(L,n,true,Gamma{j}(:,i));
      f(j,idx) = tmp;
   end
end

F = cell(M,1); %cell of annotator responses. 

for i=1:M 
    F{i} = sparse(f(i,:),1:N,1,L,N);
end

end

