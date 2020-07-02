clearvars; close all; clc;
addpath(genpath(pwd));

kl_optim_method=1;

%No of iterations
iter =1;


u1=zeros(1,iter);
u2=zeros(1,iter);
u3=zeros(1,iter);
u4=zeros(1,iter);

u1_e=zeros(1,iter);
u2_e=zeros(1,iter);
u3_e=zeros(1,iter);
u4_e=zeros(1,iter);

rec_error=zeros(1,iter);
rec_error1=zeros(1,iter);


t1 = zeros(1,iter);
t2 = zeros(1,iter);




for s=1:iter
%Problem parameters
N = 10000; %number of data.
M = 25; %number of annotators.
K = 3; %number of classes. 
I= K*ones(1,M);%vector of prior probabilities. MUST BE COLUMN VECTOR
pi_vec = ones(K,1)/K;
prior_flag=0;

disp('Generating synthetic data');
y = randsample(K,N,true,pi_vec); %generate ground-truth labels.
Gamma = generate_confusion_mat(M,K,-4); %generate confusion matrices.
p_obs = 0.2*ones(M,1); %percentage of data each annotator provides labels for. 
[F,f] = generate_annotator_labels(Gamma,y,p_obs); %generate annotator estimates.


%main part of the proposed approach starts here.
[M_tens,M_mat,mean_vecs,params.M_tens_val,params.M_mat_val,N_valid_tens,N_valid_mat] = calc_annotator_moments(F,[2]);

disp('Estimating annotator confusion matrices using GreedyOpt Method');
a = tic;
%main part of the proposed approach starts here.
[M_tens,M_mat,mean_vecs,params.M_tens_val,params.M_mat_val,N_valid_tens,N_valid_mat] = calc_annotator_moments(F,[2]);
[Gamma_est,pi_vec_est,list_g] = EstConfMat_SPA(M_mat,K);

for k=1:M
    % Resolve permutation ambiguity
    Gamma_est_cat   = Gamma_est{k};
    Gamma_cat  = Gamma{k};
    [~,Pm] = perm2match(Gamma_est_cat,Gamma_cat);
    Gamma_est{k} = Gamma_est{k}*Pm;
end

pi_vec_est=prior_estim(prior_flag,list_g,M_mat,Gamma_est,K);

b = toc(a);t1(s)=b;
disp(['MultiSPA runtime : ',num2str(b)]);


disp('Estimating annotator confusion matrices using KL-ADMM Method');
if(kl_optim_method)
    % Run algorithm
    marg = combnk(1:M,2);           % marg defines the pairs of variables (or triples 2->3)
    marg = num2cell(marg,2);        % convert it to cell
    Y=get_second_order_stat(M_mat,marg);
            
    opts = {}; opts.marg = marg; opts.max_iter = 100; opts.tol_impr = 1e-6;
    opts.A0 = Gamma_est; 
    opts.l0 = pi_vec;
    I=K*ones(1,M);
    a = tic;
    [Gamma_KL,pi_vec_KL,Out] = N_CTF_AO_KL(Y,I,K,opts);

    
    % Resolve permutation ambiguity
    A_est_cat   = concatenate(Gamma_KL);
    A_true_cat  = concatenate(Gamma);
    [~,Pm] = perm2match(A_est_cat,A_true_cat);
    for n=1:size(Gamma_KL,1)
        Gamma_KL{n} = Gamma_KL{n}*Pm;
    end
    pi_vec_KL = Pm*pi_vec_KL;
    b = toc(a);t2(s)=b;
    disp(['KL-ADMM runtime : ',num2str(b)]);
    
end




% Compute error
for n=1:M
    rel_factor_error = (norm(Gamma_est{n}(:) - Gamma{n}(:))^2)/K;
    rec_error(s) =  rec_error(s) + rel_factor_error/M;
end

if(kl_optim_method)
    % Compute error
    for n=1:M
        rel_factor_error = (norm(Gamma_KL{n}(:) - Gamma{n}(:))^2)/K;
        rec_error1(s) =  rec_error1(s) + rel_factor_error/M;
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('------------Greedy Method----------------------');
disp(['Mean Estimation Error: ',num2str(rec_error(s))]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(kl_optim_method)
    disp('------------KL-ADMM Method ----------------------');
    disp(['Mean Estimation Error: ',num2str(rec_error1(s))]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% bring Gamma cells into proper form for label estimation
%%%%%%%%%%%%%%%%% function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gamma_tensor = zeros(K,K,M);
Gamma_hat_tensor = zeros(K,K,M);
F_tensor = zeros(K,N,M);
for i=1:M
    Gamma_tensor(:,:,i) = Gamma{i};
    Gamma_hat_tensor(:,:,i) = Gamma_est{i};
    F_tensor(:,:,i) = F{i};
end


if(kl_optim_method)
    Gamma_tensor2 = zeros(K,K,M);
    Gamma_hat_tensor2 = zeros(K,K,M);
    F_tensor2 = zeros(K,N,M);
    for i=1:M
        Gamma_tensor2(:,:,i) = Gamma{i};
        Gamma_hat_tensor2(:,:,i) = Gamma_KL{i};
        F_tensor2(:,:,i) = F{i};
    end
end

valid_index = find(y>0);



%%%%%%%%%%%%%%%%%%%% Compute error rates %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('-----------------------------------------------------------------');
idx = label_estimator(Gamma_hat_tensor,F_tensor,'MAP',pi_vec_est); %Estimate labels using our method
u = find(idx(valid_index)~=y(valid_index));
u1(s) = length(u);
u1_e(s) = length(u)/N;
%disp(['MAP - Number of errors ',num2str(length(u)), ' Ratio: ',num2str(length(u)/N)]);

idx_ML = label_estimator(Gamma_hat_tensor,F_tensor,'ML'); %Estimate labels using our method
u = find(idx_ML~=y);
u2(s) = length(u);
u2_e(s) = length(u)/N;
%disp(['ML - Number of errors ',num2str(length(u_ML)), ' Ratio: ',num2str(length(u_ML)/N)]);


if(kl_optim_method)
    idx = label_estimator(Gamma_hat_tensor2,F_tensor2,'MAP',pi_vec_KL); %Estimate labels using our method
    u = find(idx(valid_index)~=y(valid_index));
    u3(s) = length(u);
    u3_e(s) = length(u)/N;
    %disp(['MAP - Number of errors ',num2str(length(u)), ' Ratio: ',num2str(length(u)/N)]);

    idx_ML = label_estimator(Gamma_hat_tensor2,F_tensor2,'ML'); %Estimate labels using our method
    u = find(idx_ML~=y);
    u4(s) = length(u);
    u4_e(s) = length(u)/N;
    %disp(['ML - Number of errors ',num2str(length(u_ML)), ' Ratio: ',num2str(length(u_ML)/N)]);
end

end

disp(['MAP - NMF - Number of errors ',num2str(ceil(mean(u1))), ' Ratio: ',num2str(mean(u1_e))]);
disp(['ML - NMF - Number of errors ',num2str(ceil(mean(u2))), ' Ratio: ',num2str(mean(u2_e))]);
if(kl_optim_method)
    disp(['MAP - KL-ADMM - Number of errors ',num2str(ceil(mean(u3))), ' Ratio: ',num2str(mean(u3_e))]);
    disp(['ML - KL-ADMM - Number of errors ',num2str(ceil(mean(u4))), ' Ratio: ',num2str(mean(u4_e))]);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('------------Greedy Method----------------------');
disp(['Mean Estimation Error: ',num2str(mean(rec_error))]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(kl_optim_method)
    disp('------------KL-ADMM Method ----------------------');
    disp(['Mean Estimation Error: ',num2str(mean(rec_error1))]);
end

disp('-----------------------------------------------------------------');
disp('-----------------------------------------------------------------');
disp('------------After all iterations: run-time----------------------');

disp(['MultiSPA - runtime(s) ',num2str(mean(t1))]);

if(kl_optim_method)
    disp(['MultiSPA-KL - runtime(s) ',num2str((mean(t2)))]);
end
