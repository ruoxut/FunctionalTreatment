function [Loss,d_opt] = CV_OR(t,Z,X,Y,L) 
%L-fold cross validation to choose the number of basis functions d_opt 
%used in the outcome regression method.

% Input: 
% t: 1*t_obs time interval;
% Z: n*t_num functional treatment; 
% X: n*p covariate;
% Y: n*1 outcome vector; 
% L: L-fold cross validation;

% Output:
% Loss: CV loss;
% d_opt: the number of basis functions used in estimating causal effects.

[xi,phi,lambda,mu_Z,~] = FPCA(t,Z,99);

n = length(Y);

if nargin < 5
    L = 10;
end

d_max = size(xi,1);
d_v = 1:d_max;  
Loss = zeros(size(d_v));

perm = randperm(n); 
xi = xi(:,perm);
X = X(perm,:);
Y = Y(perm);

ind_part = round(linspace(0,n,L+1));

parfor k = 1:length(d_v)
    d = d_v(k);
    for i = 1:L 
        ind_out = (ind_part(i)+1):ind_part(i+1);
        xi_out = xi(:,ind_out);
        X_out = X(ind_out,:);
        Y_out = Y(ind_out);
                
        xi_data = xi;
        X_data = X;
        Y_data = Y; 
        xi_data(:,ind_out) = [];
        X_data(ind_out,:) = [];
        Y_data(ind_out) = []; 
        
        [a_data,b_data,b_X_data,~] = FLR_mixed_BF(t,X_data,xi_data,lambda,phi,mu_Z,Y_data,d);

        for j = 1:length(ind_out)                              
            Loss(k) = Loss(k) +  (Y_out(j) - a_data - b_data'*(trapz(t,phi(1:d,:).*mu_Z,2)+xi_out(1:d,j)) - X_out(j,:)*b_X_data ).^2;
        end
    end
end

[~,ind] = min(Loss);
d_opt = d_v(ind);

end




