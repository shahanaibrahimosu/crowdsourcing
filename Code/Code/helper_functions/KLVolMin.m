function [ A,S,it,w_vec,obj] = KLVolMin( X,K,varargin)

% This criterion is powerful in HU. And it is also very useful in other
% topics such as text mining

% criterion:
%            min_{A,S} (1/2)*||X-AS||_{2,p}^p + lambda*logdet(A^t * A+eps*I)
%            subject to          1^t S = 1^t, S>=0

% input
%    X:  Data, ideally X = AS
%    K:  model order
%    lambda: regularization parameter

% contact Xiao Fu xfu@umn.edu for reproting problems

if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end

%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------

p = 1;
MaxIt = 1000;
lambda = 1;
use_ini = 'no';
nonneg_A = 'no';
solver_prox = 'FISTA'; % one step Nesterov gradient % can also be 'ISTA' (one step gradient projection) and 'ADMM' (exact solve)
A_ini = [];
S_ini = [];
Normal_tag = 'no';
U = 0;
%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'P'
                p = varargin{i+1};
            case 'LAMBDA'
                lambda = varargin{i+1};
            case 'MAXIT'
                MaxIt = varargin{i+1};
            case 'USE_INI'
                use_ini = varargin{i+1};
            case 'NONNEG_A'
                nonneg_A = varargin{i+1};
            case 'SOLVER_PROX'
                solver_prox = varargin{i+1};
            case 'VOL_REG'
                vol_reg = varargin{i+1};
            case 'A_INI'
                A_ini = varargin{i+1};
            case 'S_INI'
                S_ini = varargin{i+1};
            case 'NORMALIZE'
                Normal_tag = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end


%% determine initialization
[M,L]=size(X);
switch  use_ini
    case 'yes'
        A = A_ini;
        S = S_ini;
    case 'no'
      [ A,S]=nnmf(X,K);
end

switch  Normal_tag
    case 'yes'
        normalize = 1;
    case 'no'
        normalize = 0;
end

F = eye(K);
w_vec = ((ones(1,L)));

epsi = 1e-12;
ee = 1e-6;
A_old = A;
tk = 1;





for it = 1:MaxIt

% it
    %-------------  solve the subproblem w.r.t. S ------------------
    
    ev = eig(A'*A);
    
    Lip = max(ev);

    switch solver_prox 
        case 'FISTA'
        if it ==1
            S_new = Prox_S(X,A,S,Lip,normalize);
        elseif it>=2
            [S_new,tk] = Prox_FISTA(X,A,S,S_old,Lip,tk,normalize);
        end
        S_old = S;
        S = S_new;   
        case 'ISTA';
        [S] = Prox_S(X,A,S,Lip,normalize);
        case 'ADMM'
        [S,U]= ADMM_Theta(A,X,S,U,K,ones(1,L));
    end
         
     %--------------- solve the subproblem w.r.t. A -----------------
     
     
    S_temp  = bsxfun(@times, S, w_vec);



    
    X_temp  = bsxfun(@times, X, w_vec);
    
    
    
    
    Y = S_temp*S'+lambda*F;
    switch nonneg_A
        case 'yes'
            
            
            mu = norm(Y,2);
            A = A - (1/mu)*( A*Y - X_temp*S');
            A = max(A,0);
            
        case 'no'
            A = X_temp*S'/Y;
    end
    %-------------  update F ---------------------------------------
    
    F = inv(A'*A+ee*eye(K));
            
    
    %--------------  update W --------------------------------------
    
    if p~=2
        w_vec = (p/2)*(sum((X-A*S).^2)+epsi).^((p-2)/2);
    end
    
     
    
    
    %------------- calculate the objective function ----------------
    
    
   
    obj(it) = (1/2)*sum((sum((X-A*S).^2)+epsi).^(p/2)) + lambda*(log(det(A'*A+ee*eye(K))));
    
    if it>1&&abs(obj(it)-obj(it-1))<1e-15 % sqrt(sum(sum((S_old - S).^2)))<1e-6 % 
        break
    end

   

end


end

function [S,tk_plus_one ]=Prox_FISTA(X,A,S,S_old,L,tk,normalize)



tk_plus_one = (1+sqrt(1+4*tk^2))/2;

Y = S + (tk-1)/tk_plus_one*(S-S_old);


S = Y -(1/L)*(-A'*X + A'*A*Y);
if normalize ==1
    S  = bsxfun(@rdivide, S, sum(abs(X))+eps);
    ST = SimplexProj(S.');
    S = ST';
    S  = bsxfun(@times, S, sum(abs(X))+eps);
else
    ST = SimplexProj(S.');
    S = ST';
    
end

end




function [theta,U]= ADMM_Theta(B,Y,theta,U,K,TrR)
rho = 1;
Big = [2*B'*B + rho*eye(K), ones(K,1);ones(1,K),0];
inverseBig = inv(Big);

[K,M]=size(theta);
X = theta;
% U = 0;
Z = 0;
Maxiter = 200;
ob=[];
for i=1:Maxiter
    XX =inverseBig*[2*B'*Y+rho*(Z-U);TrR];
    X = XX(1:K,:);
    Z = max(X + U,0);
    U = U + X - Z;
    if sum(sqrt(sum((X-Z).^2)))<=1e-5;
        break;
    end
    
    
end


theta = X;
end

function [S]=Prox_S(X,A,S,L,normalize)



S = S -(1/L)*(-A'*X + A'*A*S);
if normalize ==1
    S  = bsxfun(@rdivide, S, sum(abs(X))+eps);
    
    ST = SimplexProj(S.');
    S = ST';
    
    S  = bsxfun(@times, S, sum(abs(X))+eps);
else
    ST = SimplexProj(S.');
    S = ST';
    
end


end





function X = SimplexProj(Y) 
% Projection to a simplex; the codes are from the reference [41]
% W. Wang and M. A. Carreira-Perpin?n, ?Projection onto the probability
% arXiv preprint arXiv:1309.1541v1, 2013.
[N,D] = size(Y);
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
end
