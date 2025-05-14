function [xi,phi,lambda,mu_Z,d] = FPCA(t,Z,FVE)
% Functional principal component analysis
% Input:
% t: 1*t_obs time interval;
% Z: n*t_obs covariate matrix;
% FVE: fraction of variance explained to select the number of PC
% Output:
% xi: d*n PC scores;
% phi: d*t_obs PC basis functions;
% lambda: vector of eigenvalues corresponding to phi;
% mu_Z: mean function of Z;
% d: number of PCs selected by FVE.

if length(t) ~= size(Z,2)
    error('Input dimensions do not match.')
end

n = size(Z,1);
mu_Z = mean(Z,1);
Z_cen = Z - mu_Z;
Cov_Z = Z_cen' * Z_cen / n;
[phi,lambda,expd] = pcacov(Cov_Z);
norm_phi = sqrt(trapz(t,phi.^2,1));       
phi = phi ./ norm_phi;
lambda = norm_phi'.^2 .* lambda; 

phi = phi'; % Each row is an eigenfunction
 
% d selected by FVE 
d = 1;
s = expd(1);
while s<FVE
    d = d+1;
    s = s+expd(d);
end

xi = zeros(d,n); 
for i = 1:n
    xi_i = trapz(t,Z_cen(i,:).*phi(1:d,:),2);
    xi(:,i) = xi_i;
end
 
phi = phi(1:d,:);
 
end

