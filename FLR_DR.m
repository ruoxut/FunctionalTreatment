function [a,b,phi] = FLR_DR(t,X,Z,Y,d_opt,n_h_opt,k_2_opt)
% The doubly robust estimator combining the outcome regression estimator and
% functional stablized weight estimator.

% Input: 
% t: 1*t_obs time interval;
% X: n*p p-dimensional covariates;
% Z: n*t_obs covariate matrix;  
% Y: n*1 outcome vector; 
% d_opt: the number of PC basis;
% n_h_opt: an integer as proxy of bandwidth h;
% k_2_opt: 1 + k_2*dim of X = k in the paper, the number of basis for X;


% Output:
% a: scalar intercept;
% b: d*1 slope coefficient vector;
% phi: d*t_obs basis functions; 

if iscolumn(t)
    t = t';
end

if isrow(Y)
    Y = Y';
end

if length(t) ~= size(Z,2) || length(Y) ~= size(Z,1) || size(X,1) ~= size(Z,1)
    error('Input dimensions do not match.')
end

% FPCA
n = length(Y);
mu_Z = mean(Z,1);
Z_cen = Z - mu_Z;
Cov_Z = Z_cen' * Z_cen / n;
[phi,lambda,~] = pcacov(Cov_Z);
norm_phi = sqrt(trapz(t,phi.^2,1));       
phi = phi ./ norm_phi;
lambda = norm_phi'.^2 .* lambda; 
phi = phi'; % Each row is an eigenfunction
phi = phi(1:d_opt,:);

xi = zeros(d_opt,n); 
for i = 1:n
    xi_i = trapz(t,Z_cen(i,:).*phi,2);
    xi(:,i) = xi_i;
end
  
% Outcome regression estimator
[a_OR,b_OR_coe,b_X_OR,~] = FLR_mixed_BF(t,X,xi,lambda,phi,mu_Z,Y,d_opt); 
b_OR = sum(b_OR_coe.*phi,1);

% Functional stablized weight  
[pi_hat_NP,~,~] = weight_con_LOO(t,Z,X,n_h_opt,k_2_opt); 

% Regression estimation   
Y_DR = ( Y - (a_OR + trapz(t,b_OR.*Z,2) + X*b_X_OR) ).*pi_hat_NP ...
        + a_OR + trapz(t,b_OR.*Z,2) + mean(X*b_X_OR);
Y_bar = mean(Y_DR);
Z_bar = trapz(t,phi.*mu_Z,2);

b = 1./lambda(1:d_opt,1) .* (xi * (Y_DR-Y_bar) ./ n);
a = Y_bar - b'*Z_bar;

end

