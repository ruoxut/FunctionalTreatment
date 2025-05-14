function [pi_hat,pi_hat_mat,Ker_Z,vmat_std,exp_vv_half_inv,h] = weight_con_LOO(t,Z,X,n_h,k_2)
% Estimation of the weights pi 

% Input: 
% t: 1*t_num time points;
% Z: n*t_num functional treatment; 
% X: n*p covariate; 
% n_h: an integer as proxy of bandwidth h;
% k_2: 1 + k_2*dim of X = k in the paper, the number of basis for X;

% Output;
% pi_hat: n*1 estimated weights.
% pi_hat_mat: matrix form of pi_hat;
% Ker_Z: n-1*n kernel matrix;
% vmat_std: n*(1+k_2*p) standardized basis matrix;
% exp_vv_half_inv：standardization matrix;
% h: the acutual used bandwidth.
 
if size(X,1) ~= size(Z,1)
    error('Input dimensions do not match.')
end
 
n = size(X,1);  

% Standardisation 
X = 2.*(X - min(X,[],1))./(max(X,[],1) - min(X,[],1)) - 1;

x_eva = linspace(-1,1,200)';
Le_poly = zeros(length(x_eva),k_2);
for i = 1:size(Le_poly,2)
    Le_poly(:,i) = legendreP(i,x_eva);
end

p = size(X,2);
vmat = ones(n,1+k_2*p);
for k = 1:k_2
    vmat(:,(2+(k-1)*p):(1+k*p)) = interp1(x_eva,Le_poly(:,k),X); 
end 

% Remove linearly dependent columns of vmat
[~,p_res] = rref(vmat);
vmat = vmat(:,p_res);

% Orthonormalize basis matrix 
exp_vv =  vmat' * vmat / n;               % (1+k_2*p) x (1+k_2*p)
exp_vv_half = chol(exp_vv, 'lower');
exp_vv_half_inv = exp_vv_half^(-1);
vmat_std = vmat * exp_vv_half_inv'/sqrt(n);
 
% Kernel regression
Ker_Z = zeros(n-1,n); 
Normsqr = zeros(n-1,n);
delta_t = mean(diff(t));

for i = 1:n
    Z_data = Z;
    Z_data(i,:) = [];
    %Normsqr_i = trapz(t,(Z_data-Z(i,:)).^2,2);
    Normsqr(:,i) = sum(delta_t.*(Z_data-Z(i,:)).^2,2);
    %[~,I] = sort(Normsqr_i);
    %h = 2.*sqrt(Normsqr_i(I(n_h)));
end
Normsqr_vec = mean(Normsqr,2);
Normsqr_vec = sort(1.5.*Normsqr_vec);

h = sqrt(Normsqr_vec(n_h));
for i = 1:n
    Ker_Z(:,i) = exp(- Normsqr(:,i) ./ h^2) ; 
end

ini = zeros(1,size(vmat,2));%+1e-10.*random('Normal',0,1,[1+k_1 1+k_2*p]);
options = optimoptions('fminunc','MaxIterations',1e4,'MaxFunctionEvaluations',size(ini,1)*size(ini,2)*1e3,...
                       'StepTolerance',1e-10,'Display','off');%,'Algorithm','trust-region','SpecifyObjectiveGradient',true);

pi_hat_mat = zeros(n,n);
parfor i = 1:n
    vmat_std_data = vmat_std;
    vmat_std_data(i,:) = [];
    fun = @(gam) Obj(gam,Ker_Z(:,i),vmat_std_data);
    GammaHat = fminunc(fun,ini,options); 
    pi_hat_mat(i,:) = exp(-GammaHat*vmat_std' - 1);    % derivative of rho
end

pi_hat = diag(pi_hat_mat);

end


function [f,g] = Obj(gam,Ker_Z_i,vmat)

norm_K = sum(Ker_Z_i);
meanvmat = mean(vmat,1);
 
term1 = (-exp(-gam * vmat' - 1)) * Ker_Z_i / norm_K;
term2 = gam * meanvmat';

f = -term1 + term2; % Function -G(Lambda)

term1_g = zeros(1,length(gam));
for i = 1:length(gam)
    term1_g(1,i) = exp(-gam * vmat' - 1) * (vmat(:,i).*Ker_Z_i) / norm_K;
end

term2_g = meanvmat;

g = -term1_g + term2_g; % Gradient of -G(Lambda)

end
 



