function [a,b,phi] = FLR(t,Z,Y,d)
% Functional linear regression with functional covariate Z and scalar
% outcome Y.

% Input: 
% t: 1*t_obs time interval;
% Z: n*t_obs covariate matrix;
% Y: n*1 outcome vector;

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

if length(t) ~= size(Z,2) || length(Y) ~= size(Z,1)
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
  
xi = zeros(d,n); 
for i = 1:n
    xi_i = trapz(t,Z_cen(i,:).*phi(1:d,:),2);
    xi(:,i) = xi_i;
end
 

% Regression estimation   
Y_bar = mean(Y);
Z_bar = trapz(t,phi(1:d,:).*mu_Z,2);

b = 1./lambda(1:d,1) .* (xi * (Y-Y_bar) ./ n);
a = Y_bar - b'*Z_bar;

phi = phi(1:d,:);

end
