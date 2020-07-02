function [err_vec] = Gamma_error(Gamma,Gamma_hat,norm_type,display)
%Gamma_ERROR Function to estimate the error of the estimated Gamma matrices
%w.r.t. the actual ones.
%           Input: Gamma - Mx1 cell containing the LxL actual Gamma matrices
%              Gamma_hat - Mx1 cell containing the LxL estimated Gamma matrices
%            norm_type - type of norm to use for error.
%              display - 0/1 scalar indicating whether to show a graph with
%              the errors. If set to one a bar graph will appear.
%          Output: err_vec - Mx1 vector containing the errors for each one
%          of the LxL Gamma matrices.
%   Panagiotis Traganitis. traga003@umn.edu
if nargin < 3
    norm_type = 'fro';
end
if nargin < 4
    display = 0;
end

M = length(Gamma);
err_vec = zeros(M,1);
for i=1:M
    err_vec(i) = norm(Gamma{i} - Gamma_hat{i},norm_type)/norm(Gamma{i},norm_type);
end

if display
    figure;
    bar(err_vec)
    if ischar(norm_type)
        title(['Gamma ',norm_type,' norm estimation error']);
    else
        title(['Gamma ', num2str(norm_type), ' norm estimation error']);
    end
end
end

