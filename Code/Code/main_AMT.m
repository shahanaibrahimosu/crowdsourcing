clearvars; close all; clc;
addpath(genpath(pwd));

tensor_method=0;
KLmethod=1;
Spectral_Method=0;
KOS_method=0;
EigenRatio_method=0;
GhostSVD_method=0;
Minmax_entropy=0;
iter =1;

u1=zeros(1,iter);
u2=zeros(1,iter);
u3=zeros(1,iter);
u4=zeros(1,iter);
u5=zeros(1,iter);
u6=zeros(1,iter);
u7=zeros(1,iter);
u8=zeros(1,iter);
u9=zeros(1,iter);
u10=zeros(1,iter);
u11=zeros(1,iter);
u12=zeros(1,iter);
u13=zeros(1,iter);
u14=zeros(1,iter);
u15=zeros(1,iter);
u16=zeros(1,iter);
u17=zeros(1,iter);
u18=zeros(1,iter);

t1 = zeros(1,iter);
t2 = zeros(1,iter);
t3 = zeros(1,iter);
t4 = zeros(1,iter);
t5 = zeros(1,iter);
t6 = zeros(1,iter);
t7 = zeros(1,iter);
t8 = zeros(1,iter);
t9 = zeros(1,iter);
t10 = zeros(1,iter);


u1_e=zeros(1,iter);
u2_e=zeros(1,iter);
u3_e=zeros(1,iter);
u4_e=zeros(1,iter);
u5_e=zeros(1,iter);
u6_e=zeros(1,iter);
u7_e=zeros(1,iter);
u8_e=zeros(1,iter);
u9_e=zeros(1,iter);
u10_e=zeros(1,iter);
u11_e=zeros(1,iter);
u12_e=zeros(1,iter);
u13_e=zeros(1,iter);
u14_e=zeros(1,iter);
u15_e=zeros(1,iter);
u16_e=zeros(1,iter);
u17_e=zeros(1,iter);
u18_e=zeros(1,iter);


for s=1:iter

disp('Load the real Data');
[F,f,F_orig,f_orig,y,K,M,N,conf_mat]    = LoadDataset_Dog();%LoadDataset_Dog()
%LoadDataset_RTE()%LoadDataset_bluebird()%LoadDataset_trec()


a = tic;
%main part of the proposed approach starts here.
[M_tens,M_mat,mean_vecs,params.M_tens_val,params.M_mat_val,N_valid_tens,N_valid_mat] = calc_annotator_moments(F,[2]);
disp('Estimating annotator confusion matrices using MultiSPA Method');
[Gamma_est,pi_vec_est,list_g] = EstConfMat_SPA(M_mat,K);
[~,Gamma_est] = getPermutedMatrix(Gamma_est,list_g);
b = toc(a);t1(iter)=b;
disp(['MultiSPA runtime : ',num2str(b)]);





if(tensor_method)
    %AO-ADMM Algorithm parameters
    params.inner_max_iter = 10; %inner ADMM max iterations
    params.outer_max_iter = 100; %outer loop max iterations
    params.outer_tol = 1e-12; %outer loop tolerance
    params.inner_tol = 1e-8; %inner loop tolerance
    params.init = 0; %Algorithm initialization. Set to 1 for better than random.
    params.display = 0;
    
    a = tic;
    %main part of the proposed approach starts here.
    [M_tens,M_mat,mean_vecs,params.M_tens_val,params.M_mat_val,N_valid_tens,N_valid_mat] = calc_annotator_moments(F,[1,2,3]);
    disp('Estimating annotator confusion matrices using ADMM');
    [Gamma_hat,pi_vec_hat] = EstConfMat_AO_ADMM(mean_vecs,M_mat,M_tens,params);
    %Resolve permutaion ambiguity
    [Pm1,Gamma_hat] = getPermutedMatrix(Gamma_hat,1:M);
    b = toc(a);t2(iter)=b;
    disp(['TensorADMM runtime : ',num2str(b)]);
end


if(KLmethod)
    % Run algorithm
    marg = combnk(1:M,2);           % marg defines the pairs of variables (or triples 2->3)
    marg = num2cell(marg,2);        % convert it to cell
    Y=get_second_order_stat(M_mat,marg);    
            
    opts = {}; opts.marg = marg; opts.max_iter = 10; opts.tol_impr = 1e-6;
    Gamma_est=algorithm_init(Gamma_est,f_orig,N,K,M);
    opts.A0 = Gamma_est;
    opts.l0 = pi_vec_est;
    I=K*ones(1,M);
    disp('Estimating annotator confusion matrices using MultiSPA-KL Method');
    a = tic;
    [Gamma_KL,pi_vec_KL,Out] = N_CTF_AO_KL(Y,I,K,opts);
    b = toc(a);t3(iter)=b;
    disp(['MultiSPA-KL runtime : ',num2str(b)]);
end


%%%%%%%%%%%%%%%%% bring Gamma cells into proper form for label estimation
%%%%%%%%%%%%%%%%% function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M_mod = length(list_g);
list_g1= list_g;
Gamma_hat_tensor = zeros(K,K,M_mod);
F_tensor = zeros(K,N,M_mod);

j=1;
for i=1:M_mod
    Gamma_hat_tensor(:,:,i) = Gamma_est{list_g1(i)};
    F_tensor(:,:,i) = F{list_g1(i)};
    j=j+1;
end



if(tensor_method)
    Gamma_hat_tensor1 = zeros(K,K,M);
    F_tensor1 = zeros(K,N,M);
    for i=1:M
        Gamma_hat_tensor1(:,:,i) = Gamma_hat{i};
        F_tensor1(:,:,i) = F{i};
    end
end

if(KLmethod)
    M_mod = M;
    list_g2 = 1:M;
    Gamma_hat_tensor2 = zeros(K,K,M);
    F_tensor2 = zeros(K,N,M);
    for i=1:M_mod
        Gamma_hat_tensor2(:,:,i) = Gamma_KL{list_g2(i)};
        F_tensor2(:,:,i) = F{list_g2(i)};
    end
end
valid_index = find(y>0);
N_valid=length(valid_index);
%%%%%%%%%%%%%%%%%%%% Compute error rates %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('-----------------------------------------------------------------');
idx = label_estimator(Gamma_hat_tensor,F_tensor,'MAP',pi_vec_est); %Estimate labels using our method
u = find(idx(valid_index)~=y(valid_index));
u1(s) = length(u);
u1_e(s) = length(u)/N_valid;
disp(['MultiSPA-MAP - Number of errors ',num2str(u1(s)), ' Ratio: ',num2str(u1_e(s))]);

idx = label_estimator(Gamma_hat_tensor,F_tensor,'ML'); %Estimate labels using our method
u = find(idx(valid_index)~=y(valid_index));
u2(s) = length(u);
u2_e(s) = length(u)/N_valid;
disp(['MultiSPA-ML - Number of errors ',num2str(u2(s)), ' Ratio: ',num2str(u2_e(s))]);


F_tensor_MV=zeros(K,N,M);
for i=1:M
    F_tensor_MV(:,:,i)=F_orig{i};
end
[u3_e(s)] = majority_voting_label(F_tensor_MV,y,valid_index);
u3(s) = round(u3_e(s)*length(valid_index));
disp(['Majority vote - Number of errors ',num2str(u3(s)), ' Ratio: ',num2str(u3_e(s))]);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run EM with initialization provided from
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% our method 
% 
 
Nround = 10; %number of EM iterations
a = tic;
[A] = convert_for_comp(f_orig);
Z = zeros(N,K,M);
for i = 1:size(A,1)
 Z(A(i,1),A(i,3),A(i,2)) = 1;
end
Gamma_hat_tensor_EM = zeros(K,K,M);
F_tensor_EM = zeros(K,N,M);
for i=1:M
    Gamma_hat_tensor_EM(:,:,i) = Gamma_est{i};
    F_tensor_EM(:,:,i) = F{i};
end

[error_EM_predict] = run_EM(Gamma_hat_tensor_EM,Z,y,valid_index,Nround);
u6(s) = ceil(error_EM_predict(end)*N_valid);
u6_e(s) = error_EM_predict(end);
disp(['MultiSPA-EM- Number of errors ',num2str(ceil(error_EM_predict(end)*N)), ' Ratio: ',num2str(error_EM_predict(end))]);
b = toc(a);t4(iter)=b;
disp(['MultiSPA-EM runtime : ',num2str(b)]);

if(tensor_method)
    idx = label_estimator(Gamma_hat_tensor1,F_tensor_MV,'MAP',pi_vec_hat); %Estimate labels using our method
    u = find(idx(valid_index)~=y(valid_index));
    u8(s) = length(u);
    u8_e(s) = length(u)/N_valid;
    disp(['MAP - Number of errors ',num2str(u8(s)), ' Ratio: ',num2str(u8_e(s))]);

    idx = label_estimator(Gamma_hat_tensor1,F_tensor_MV,'ML'); %Estimate labels using our method
    u = find(idx(valid_index)~=y(valid_index));
    u9(s) = length(u);
    u9_e(s) = length(u)/N_valid;
    disp(['ML - Number of errors ',num2str(u9(s)), ' Ratio: ',num2str(u9_e(s))]);
end
 
if(KLmethod)
    idx = label_estimator(Gamma_hat_tensor2,F_tensor2,'MAP',pi_vec_KL); %Estimate labels using our method
    u = find(idx(valid_index)~=y(valid_index));
    u12(s) = length(u);
    u12_e(s) = length(u)/N_valid;
    disp(['MAP - Number of errors ',num2str(u12(s)), ' Ratio: ',num2str(u12_e(s))]);

    idx = label_estimator(Gamma_hat_tensor2,F_tensor2,'ML'); %Estimate labels using our method
    u = find(idx(valid_index)~=y(valid_index));
    u13(s) = length(u);
    u13_e(s) = length(u)/N_valid;
    disp(['ML - Number of errors ',num2str(u13(s)), ' Ratio: ',num2str(u13_e(s))]);
 end

%%%%%%%%%%%%%%%%%%%%% Run EM initialized with majority vote
a = tic;
[A] = convert_for_comp(f_orig);
Z = zeros(N,K,M);
for i = 1:size(A,1)
 Z(A(i,1),A(i,3),A(i,2)) = 1;
end
error_EM_MV_predict = run_EM_MV(Z,y,valid_index,Nround);
u7(s) = ceil(error_EM_MV_predict(end)*N_valid);
u7_e(s) = error_EM_MV_predict(end);
b=toc(a);t5(iter)=b;
disp(['EM MV - Number of errors ',num2str(ceil(error_EM_MV_predict(end)*N)), ' Ratio: ',num2str(error_EM_MV_predict(end))]);
disp(['EM MV runtime : ',num2str(b)]);

if(Spectral_Method)
    a=tic;
    [A] = convert_for_comp(f_orig);
    Z = zeros(N,K,M);
    for i = 1:size(A,1)
        Z(A(i,1),A(i,3),A(i,2)) = 1;
    end
    error_EM_predict = run_EM_Spectral(Z,y,valid_index,Nround);
    u14(s) = ceil(error_EM_predict(end)*N_valid);
    u14_e(s) = error_EM_predict(end);
    b=toc(a);t6(iter)=b;
    disp(['EM MV - Number of errors ',num2str(ceil(error_EM_predict(end)*N)), ' Ratio: ',num2str(error_EM_predict(end))]);
    disp(['Spectral-EM runtime : ',num2str(b)]);
end
if(KOS_method)
    a=tic;
    [A] = convert_for_comp(f_orig);
    [error_KOS] = KOS(A,y,valid_index,N,K,M);
    u15(s) = ceil(error_KOS(end)*N_valid);
    u15_e(s) = error_KOS(end);
    b=toc(a);t7(iter)=b;
    disp(['KOS - Number of errors ',num2str(u15(s)), ' Ratio: ',num2str(u15_e(s))]);
    disp(['KOS runtime : ',num2str(b)]);
end
if(GhostSVD_method)
    a = tic;
    [A] = convert_for_comp(f_orig);
    [error_GhostSVD] = GhostSVD(A,y,valid_index,N,K,M);
    u16(s) = ceil(error_GhostSVD(end)*N_valid);
    u16_e(s) = error_GhostSVD(end);
    b=toc(a);t8(iter)=b;
    disp(['GhoshSVD - Number of errors ',num2str(u16(s)), ' Ratio: ',num2str(u16_e(s))]);
    disp(['GhoshSVD runtime : ',num2str(b)]);
end
if(EigenRatio_method)
    a = tic;
    [A] = convert_for_comp(f_orig);
    [error_RatioEigen] = EigenRatio(A,y,valid_index,N,K,M);
    u17(s) = ceil(error_RatioEigen(end)*N_valid);
    u17_e(s) = error_RatioEigen(end);
    b=toc(a);t9(iter)=b;
    disp(['EigenRatio - Number of errors ',num2str(u17(s)), ' Ratio: ',num2str(u17_e(s))]);
    disp(['EigenRatio runtime : ',num2str(b)]);
end
if(Minmax_entropy)
    L = f';
    true_labels=y';
    Model = crowd_model(L, 'true_labels',true_labels);
    a=tic;
    % Set parameters:
    lambda_worker = 0.25*Model.Ndom^2; lambda_task = lambda_worker * (mean(Model.DegWork)/mean(Model.DegTask)); % regularization parameters
    opts={'lambda_worker', lambda_worker, 'lambda_task', lambda_task, 'maxIter',50,'TOL',5*1e-3','verbose',1};
    % 1. Categorical minimax entropy:
    result1 =  MinimaxEntropy_crowd_model(Model,'algorithm','categorical',opts{:});
    u18_e(s)=result1.error_rate;
    u18(s) = ceil(u18_e(s)*N);
    b=toc(a);t10(iter)=b;
    disp(['MinmaxEntropy runtime : ',num2str(b)]);
end
end

disp('-----------------------------------------------------------------');
disp('-----------------------------------------------------------------');
disp('------------After all iterations: error rate----------------------');

disp(['MAP-MultiSPA - Number of errors ',num2str(ceil(mean(u1))), ' Ratio: ',num2str(mean(u1_e))]);
disp(['ML-MultiSPA - Number of errors ',num2str(ceil(mean(u2))), ' Ratio: ',num2str(mean(u2_e))]);
disp(['Majority vote - Number of errors ',num2str(ceil(mean(u3))), ' Ratio: ',num2str(mean(u3_e))]);
disp(['MultiSPA-EM - Number of errors ',num2str(ceil(mean(u6))), ' Ratio: ',num2str(mean(u6_e))]);
if(tensor_method)
    disp(['MAP-TensorADMM - Number of errors ',num2str(ceil(mean(u8))), ' Ratio: ',num2str(mean(u8_e))]);
    disp(['ML-TensorADMM - Number of errors ',num2str(ceil(mean(u9))), ' Ratio: ',num2str(mean(u9_e))]);
end
if(KLmethod)
    disp(['MAP-MultiSPA-KL - Number of errors ',num2str(ceil(mean(u12))), ' Ratio: ',num2str(mean(u12_e))]);
    disp(['ML-MultiSPA-KL - Number of errors ',num2str(ceil(mean(u13))), ' Ratio: ',num2str(mean(u13_e))]);
end
disp(['EM MV-Number of errors ',num2str(ceil(mean(u7))), ' Ratio: ',num2str(mean(u7_e))]);
if(Spectral_Method)
 disp(['Spectral-EM - Number of errors ',num2str(ceil(mean(u14))), ' Ratio: ',num2str(mean(u14_e))]);   
end
if(KOS_method)
 disp(['KOS - Number of errors ',num2str(ceil(mean(u15))), ' Ratio: ',num2str(mean(u15_e))]);   
end
if(GhostSVD_method)
 disp(['GhostSVD - Number of errors ',num2str(ceil(mean(u16))), ' Ratio: ',num2str(mean(u16_e))]);   
end
if(EigenRatio_method)
 disp(['Eigen Ratio - Number of errors ',num2str(ceil(mean(u17))), ' Ratio: ',num2str(mean(u17_e))]);   
end
if(Minmax_entropy)
 disp(['MinmaxEntropy - Number of errors ',num2str(ceil(mean(u18))), ' Ratio: ',num2str(mean(u18_e))]);       
end
disp('-----------------------------------------------------------------');
disp('-----------------------------------------------------------------');
disp('------------After all iterations: run-time----------------------');

disp(['MultiSPA - runtime(s) ',num2str(mean(t1))]);
disp(['MultiSPA-EM - runtime(s) ',num2str((mean(t4)))]);

if(tensor_method)
    disp(['TensorADMM - runtime(s) ',num2str((mean(t2)))]);
end
if(KLmethod)
    disp(['MultiSPA-KL - runtime(s) ',num2str((mean(t3)))]);
end
disp(['EM-MV - runtime(s) ',num2str((mean(t5)))]);
if(Spectral_Method)
    disp(['Spectral-EM - runtime(s) ',num2str((mean(t6)))]);
end
if(KOS_method)
    disp(['KOS - runtime(s) ',num2str((mean(t7)))]);
end
if(GhostSVD_method)
    disp(['GhoshSVD - runtime(s) ',num2str((mean(t8)))]);
end
if(EigenRatio_method)
    disp(['EigenRatio - runtime(s) ',num2str((mean(t9)))]);
end
if(Minmax_entropy)
    disp(['MinmaxEntropy - runtime(s) ',num2str((mean(t10)))]);
end