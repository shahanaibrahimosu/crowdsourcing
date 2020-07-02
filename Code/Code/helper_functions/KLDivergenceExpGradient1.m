function [alpha_est,obj_vec] = KLDivergenceExpGradient1(y,x,alpha_init)
c1 = 10^(-2);
c2 = 0.9;
eps=10^(-7);
it=1;


N=size(y,2);
F=size(alpha_init,2);


m=cell(N,1);
n=cell(N,1);
p=cell(N,1);
alpha=[];
proj = cell(N,1);
old_proj = cell(N,1);
alpha_est =zeros(N,F);
for i=1:N
    m{i} = y(:,i);
    n{i} = x;
    m{i}=m{i}+10^(-9);
    n{i}=n{i}+10^(-9);
    alpha_init=alpha_init+10^(-9);
    %n{i} = n{i}*diag(1./sum(n{i},1)); 
    alpha_init = alpha_init*diag(1./sum(alpha_init,1));  
    p{i} = alpha_init(i,:);
    alpha = [alpha p{i}];
    proj{i} = n{i}*p{i}';
end

    
old_alpha = alpha;
log_alpha = log(alpha);
old_log_alpha = log_alpha;
old_proj = proj;

new_obj = KL(m,proj);

g = cell(N,1);
grad=[];
for i=1:N
    g{i}= -(m{i}')*diag(1./proj{i})*n{i};
    grad= [grad g{i}];
end
old_grad = grad;

stepsize=0.2; %Stepsize 5 is good for N=50,I=5
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
    
    y_est=[];
    for i=1:N
        y_est= [y_est n{i}*(alpha((i-1)*F+1:i*F))'];
    end
    sum_d = sum(y_est,'all');
    D=(1/sum_d)*eye(size(x,1));
    x=D*x;
    for i=1:N
        n{i}=x;
    end
    
    
    for i=1:N
        proj{i} = n{i}*(alpha((i-1)*F+1:i*F))';
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
        g{i}= -(m{i}')*diag(1./proj{i})*n{i};
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
ret=0;
j=[];
g=[];
for k=1:N
    g = [g proj{k}];
    j = [j y{k}];
end
log_diff = log(j)-log(g);
ret=sum(j.*log_diff,'all');
end
