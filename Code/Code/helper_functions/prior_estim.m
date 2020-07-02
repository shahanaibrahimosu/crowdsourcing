function[p_vec_hat]= prior_estim(prior_flag,list_good_annot,M_mat,A_est_G,K)

if(prior_flag)
    count=0;
    p_vec_hat=zeros(K,1);
    for i=list_good_annot
        for j=list_good_annot
            if(i~= j && rank(M_mat{i,j})== K)
                count=count+1;
                A_e = khatrirao(A_est_G{j},A_est_G{i});
                temp=pinv(A_e)*M_mat{i,j}(:);
                temp(temp<0)=0;
                temp=temp./sum(temp);
                temp(isnan(temp))=0;
                p_vec_hat = p_vec_hat+temp;
            end
        end
    end
    p_vec_hat=p_vec_hat/count;
    p_vec_hat=p_vec_hat./sum(p_vec_hat);
else
    p_vec_hat = ones(K,1)/K;
end