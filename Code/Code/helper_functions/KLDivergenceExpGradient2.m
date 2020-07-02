function [alpha_est,obj_vec] = KLDivergenceExpGradient2(Y,X,alpha_init)
c1 = 10^(-3);
c2 = 0.9;
eps=10^(-7);
it=1;

M=size(Y,1);
N=size(Y{1},2);
F=size(alpha_init,2);

m=cell(N,M);
n=cell(M,1);
p=cell(N,1);

proj = cell(N,M);
old_proj = cell(N,1);
alpha_est =zeros(N,F);
for j=1:M
    n{j} = X{j};
    n{j}=n{j}+10^(-9);
    alpha=[];
    for i=1:N
        m{i,j} = Y{j}(:,i);
        m{i,j}=m{i,j}+10^(-9);
        alpha_init=alpha_init+10^(-9);
        p{i} = alpha_init(i,:);
        alpha = [alpha p{i}];
        proj{i,j} = n{j}*p{i}';
    end
end

    
old_alpha = alpha;
log_alpha = log(alpha);
old_log_alpha = log_alpha;
old_proj = proj;

new_obj = KL(m,proj);

g = cell(N,M);
grad=[];

for i=1:N
    g{i} = zeros(1,F);
    for j=1:M
        h= -(m{i,j}')*diag(1./proj{i,j})*n{j};
        g{i}=g{i}+h;
    end
    grad= [grad g{i}];
end

old_grad = grad;

stepsize=0.1; %Stepsize 5 is good for N=50,I=5
decreasing = 'False';
repeat = 'False';
gap = 1/0;
max_it=100;
obj_vec=[];

while(it<max_it)
    eta=stepsize;
    old_obj=new_obj;
    old_alpha=alpha;
%     alpha_init
%     alpha_true;
%     alpha_true_i
%     alpha;
%     stepsize
%     new_obj
%     y
%     x*alpha'
    old_log_alpha=log_alpha;
    old_proj = proj;
    
    it=it+1;
    
    %Take a step
    log_alpha = log_alpha-eta*grad;    
    
    %Compute new objective
    alpha=exp(log_alpha);
    %Normalize to project to simplex
    alpha_est = (reshape(alpha,[F,N]))';
    alpha_est = alpha_est*diag(1./sum(alpha_est,1));
    alpha     = reshape(alpha_est',[1 N*F]);
    
    for j=1:M
        y_est=[];
        for i=1:N
            y_est= [y_est n{j}*(alpha((i-1)*F+1:i*F))'];
        end
        sum_d = sum(y_est,'all');
        D=(1/sum_d)*eye(size(X{j},1));
        X{j}=D*X{j};
        n{j}=X{j};
    end
    
    
    for i=1:N
        proj{i,j} = n{j}*(alpha((i-1)*F+1:i*F))';
    end
    new_obj=KL(m,proj);
    obj_vec=[obj_vec new_obj];
    if(new_obj <eps)
        break;
    end
    
    grad_dot_deltaAlpha = dot(grad, alpha - old_alpha);
    %assert (grad_dot_deltaAlpha <= 10^(-9));
    if(~(new_obj <= old_obj + c1*stepsize*grad_dot_deltaAlpha)) %sufficient decrease
        stepsize =stepsize/ 2.0; %reduce stepsize;
        if(stepsize < 10^(-6))
            break;
        end
        alpha = old_alpha ;
        log_alpha = old_log_alpha;
        proj = old_proj;
        new_obj = old_obj;
        repeat = 'True';
        decreasing = 'True';
        continue;
    end


    %compute the new gradient
    old_grad = grad;
    grad=[];
    for i=1:N
        g{i} = zeros(1,F);
        for j=1:M
            h= -(m{i,j}')*diag(1./proj{i,j})*n{j};
            g{i}=g{i}+h;
        end
        grad= [grad g{i}];
    end
    
    

    if(~(dot(grad, alpha - old_alpha) >= c2*grad_dot_deltaAlpha) & (~ decreasing)) %curvature
        stepsize = stepsize*2.0; %increase stepsize
        alpha = old_alpha;
        log_alpha = old_log_alpha;
        grad = old_grad;
        proj = old_proj;
        new_obj = old_obj;
        repeat = 'True';
        continue;
    end

    decreasing= 'False';
    mu = [];
    for i=1:N
        s = ones(1,F)*min(grad((i-1)*F+1:i*F));
        mu =[mu s];
    end
    lam =grad-mu;
    
    gap=0;
    for i=1:N
        ind=(i-1)*F+1:i*F;
        gap = gap+dot(alpha(ind), lam(ind));
    end
    convergence = gap;
    if (convergence < eps)
        break
    end
       
end

for i=1:N
    alpha_est(i,:)=alpha((i-1)*F+1:i*F);
end
end


function [ret] = KL(y,proj)
N=size(y,1);
M=size(y,2);
ret=0;
for u=1:M
    j=[];
    g=[];
    for k=1:N
        g = [g proj{k,u}];
        j = [j y{k,u}];
    end
    log_diff = log(j)-log(g);
    ret=ret+sum(j.*log_diff,'all');
end
end
