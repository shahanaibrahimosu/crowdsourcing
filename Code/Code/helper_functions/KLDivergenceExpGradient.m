function [alpha_est,obj_vec] = KLDivergenceExpGradient(y,x,alpha)
c1 = 10^(-2);
c2 = 0.9;
eps=10^(-7);
it=1;


[K,N]=size(x);
mask = find(y);
y=y(mask);
x=x(mask,:);
x=x+10^(-9);
%y(find(y==0))=10^(-9);
x = x*diag(1./sum(x,1));
y_sum=sum(y,1);
y = y/y_sum;
alpha=alpha/sum(alpha);
alpha_init=alpha;

old_alpha = alpha;
log_alpha = log(alpha);
old_log_alpha = log_alpha;
proj = x*alpha';
old_proj = proj;

log_y = log(y);
new_obj = KL(y,log_y,proj);
y_over_proj = y/proj;
y_over_proj=y_over_proj(find(y_over_proj));
grad= -(y')*diag(1./proj)*x;%-(y_over_proj')*x;
old_grad = grad;

stepsize=0.3;
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
    
    %Normalize to project to simplex
    log_alpha = log_alpha-logsumexp(log_alpha);
    
    %Compute new objective
    alpha=exp(log_alpha);
    proj=x*alpha';
    new_obj=KL(y,log_y,proj);
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
    y_over_proj = y/proj;
    y_over_proj=y_over_proj(find(y_over_proj));
    grad=-(y')*diag(1./proj)*x;%-(y_over_proj')*x;

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
    lam = grad;
    lam =lam-min(lam);

    gap = dot(alpha, lam);
    convergence = gap;
    if (convergence < eps)
        break
    end
       
end
alpha_est =alpha;
end


function [ret] = KL(p,log_p,q)
N=length(p);
ret=0;
log_diff = log_p-log(q);
ret=dot(p,log_diff);
end
