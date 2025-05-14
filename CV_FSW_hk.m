function [Loss,n_h_opt,k_2_opt] = CV_FSW_hk(t,Z,X,Y,L,d_opt,n_h_candi,k_2_candi) 
%L-fold cross validation to choose the number of basis functions d_opt used in estimating the causal effect,
%k_2_opt used in estimating the weights.

% Input: 
% t: 1*t_obs time interval;
% Z: n*t_num functional treatment; 
% X: n*p covariate;
% Y: n*1 outcome vector; 
% L: L-fold cross validation;
% d_opt: the number of basis functions;
% n_h_candi: candidates for n_h;
% k_2_candi: candidates for k_2;

% Output:
% Loss: CV loss; 
% n_h_opt; the number of curves used in kernel regression
% k_2_opt: the number of basis functions used in estimating the
% weights.

[xi,phi,lambda,mu_Z,~] = FPCA(t,Z,99);

n = length(Y);

[n_h_v,k_2_v] = ndgrid(n_h_candi,k_2_candi);
n_h_v = n_h_v(:);
k_2_v = k_2_v(:);
Loss = zeros(size(n_h_v));

perm = randperm(n);
Z = Z(perm,:);
xi = xi(:,perm);
X = X(perm,:);
Y = Y(perm);

ind_part = round(linspace(0,n,L+1));
n_out = diff(ind_part);
n_data = n.*ones(1,L);
n_data = n_data - n_out;

x_eva = linspace(-1,1,200)';
Le_poly = zeros(length(x_eva),max(k_2_v));
for i = 1:size(Le_poly,2)
    Le_poly(:,i) = legendreP(i,x_eva);
end

parfor k = 1:length(n_h_v)
    n_h = n_h_v(k);
    k_2 = k_2_v(k);
    
    for i = 1:L 
        ind_out = (ind_part(i)+1):ind_part(i+1);
        Z_out = Z(ind_out,:);
        xi_out = xi(:,ind_out);
        X_out = X(ind_out,:);
        Y_out = Y(ind_out);
                
        Z_data = Z;
        xi_data = xi;
        X_data = X;
        Y_data = Y; 
        Z_data(ind_out,:) = [];
        xi_data(:,ind_out) = [];
        X_data(ind_out,:) = [];
        Y_data(ind_out) = []; 
                
        [pi_d_data,~,~,vmat_data_std,exp_vv_half_inv,h] = weight_con_LOO(t,Z_data,X_data,n_h,k_2);

        % Standardisation
        X_out = 2.*(X_out - min(X_out,[],1))./(max(X_out,[],1) - min(X_out,[],1)) - 1;

        p = size(X_out,2);
        vmat_out = ones(n_out(i),1+k_2*p);

        for j = 1:k_2
            vmat_out(:,(2+(j-1)*p):(1+j*p)) = interp1(x_eva,Le_poly(:,j),X_out);
        end
        
        % Remove linearly dependent columns of vmat
        [~,p_res] = rref(vmat_out);
        vmat_out = vmat_out(:,p_res);

        % Standardize basis matrix using estimates from training data
        vmat_out_std = vmat_out * exp_vv_half_inv'/sqrt(n_data(i));
 
 
        ini = zeros(1,size(vmat_out,2));%+1e-10.*random('Normal',0,1,[1+k_1 1+k_2*p]);
        options = optimoptions('fminunc','MaxIterations',1e4,'MaxFunctionEvaluations',size(ini,1)*size(ini,2)*1e3,...
                       'StepTolerance',1e-10,'Display','off');%,'Algorithm','trust-region','SpecifyObjectiveGradient',true);

        pi_d_out = zeros(n_out(i),1);
        for j = 1:n_out(i)
            Normsqr_j = trapz(t,(Z_data-Z_out(j,:)).^2,2);
            %[~,I] = sort(Normsqr_j);
            %h = 2.*sqrt(Normsqr_j(I(n_h)));
            Ker_Z_j = exp(- Normsqr_j ./ h^2) ; 
            fun = @(gam) Obj(gam,Ker_Z_j,vmat_data_std);
            GammaHat = fminunc(fun,ini,options);
            vvec_std = vmat_out_std(j,:);
            pi_d_out(j) = exp(-GammaHat*vvec_std' - 1);
        end
        
        Y_data_weighted = Y_data.*pi_d_data; 
        Y_bar = mean(Y_data_weighted);
        Z_bar = trapz(t,phi(1:d_opt,:).*mu_Z,2);
        b = 1./lambda(1:d_opt,1) .* (xi_data(1:d_opt,:) * (Y_data_weighted-Y_bar) ./ n_data(i));
        a = Y_bar - b'*Z_bar;
        
        for j = 1:length(ind_out)
            Loss(k) = Loss(k) +  (pi_d_out(j).*Y_out(j) - a - b'*(trapz(t,phi(1:d_opt,:).*mu_Z,2)+xi_out(1:d_opt,j)) ).^2;
        end
    end
end

[~,ind] = min(Loss); 
n_h_opt = n_h_v(ind);
k_2_opt = k_2_v(ind);

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
 



