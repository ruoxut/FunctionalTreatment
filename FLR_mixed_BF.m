function [a,b,b_X,phi] = FLR_mixed_BF(t,X,xi,lambda,phi,mu_Z,Y,d_opt)
% Outcome regression method using a mixed linear model and the backfitting algorithm
% with p-dimensional covariate, functional covariate Z (xi, PC scores) and scalar outcome Y.

% Input: 
% t: 1*t_obs time interval;
% X: n*p p-dimensional covariates;
% xi: d_max*n PC scores, d_max >= d_opt; 
% lambda: 
% phi: d_max*t_obs PC basis functions, d_max >= d_opt;
% mu_Z: 1*t_obs mean function of Z;
% Y: n*1 outcome vector; 
% d_opt: the number of PC of Z;

% Output:
% a: scalar intercept;
% b: d*1 slope coefficient vector;
% b_X: p*1 coefficient of X;
% phi: d*t_obs basis functions.

if iscolumn(t)
    t = t';
end

if isrow(Y)
    Y = Y';
end

if length(t) ~= size(phi,2) || length(Y) ~= size(xi,2) || size(X,1) ~= length(Y)
    error('Input dimensions do not match.')
end

n = length(Y);

xi = xi(1:d_opt,:);

% Regression estimation   
Y_bar = mean(Y);
Z_bar = trapz(t,phi(1:d_opt,:).*mu_Z,2);

b_ini = 1./lambda(1:d_opt,1) .* (xi * (Y-Y_bar) ./ n);
a_ini = Y_bar - b_ini'*Z_bar;

XI_Z = zeros(n,d_opt+1);
XI_Z(:,1) = 1;
XI_Z(:,2:d_opt+1) = xi' + trapz(t,mu_Z.*phi(1:d_opt,:),2)';

%ab_t_0 = zeros(d_opt+1,1);
%b_X_t_0 = zeros(size(X,2),1);

ab_t_1 = [a_ini;b_ini];
DM_X = (X' * X)^(-1);
Y_re_Z = Y - XI_Z * ab_t_1;
b_X_t_1 = DM_X * X' * Y_re_Z;

Y_re_X = Y - X * b_X_t_1;
Y_bar_re_X = mean(Y_re_X);
b_upd = 1./lambda(1:d_opt,1) .* (xi * (Y_re_X-Y_bar_re_X) ./ n);
a_upd = Y_bar_re_X - b_upd'*Z_bar;
ab_t_2 = [a_upd;b_upd];
Y_re_Z = Y - XI_Z * ab_t_2;
b_X_t_2 = DM_X * X' * Y_re_Z;

count = 1;
%while norm([ab_t_2;b_X_t_2]-[ab_t_1;b_X_t_1]) < norm([ab_t_1;b_X_t_1]-[ab_t_0;b_X_t_0]) && count <= 500
while norm([ab_t_2;b_X_t_2]-[ab_t_1;b_X_t_1])/norm([ab_t_1;b_X_t_1])  > 0.05 && count <= 100
%    ab_t_0 = ab_t_1;
%    b_X_t_0 = b_X_t_1;

    ab_t_1 = ab_t_2;
    b_X_t_1 = b_X_t_2;

    Y_re_X = Y - X * b_X_t_1;
    Y_bar_re_X = mean(Y_re_X);
    b_upd = 1./lambda(1:d_opt,1) .* (xi * (Y_re_X-Y_bar_re_X) ./ n);
    a_upd = Y_bar_re_X - b_upd'*Z_bar;
    ab_t_2 = [a_upd;b_upd];
    Y_re_Z = Y - XI_Z * ab_t_2;
    b_X_t_2 = DM_X * X' * Y_re_Z;
    
    count = count + 1; 
end

a = ab_t_2(1);
b = ab_t_2(2:d_opt+1);
b_X = b_X_t_2;

phi = phi(1:d_opt,:);

end

