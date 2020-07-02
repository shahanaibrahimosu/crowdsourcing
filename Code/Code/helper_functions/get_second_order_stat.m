function [Y] = get_second_order_stat(M_mat,marg)
Y  = cell(size(marg,1),1);
for i=1:size(marg,1)
    Y{i} = M_mat{marg{i}(1),marg{i}(2)};
end
end