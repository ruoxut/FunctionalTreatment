%% Code to reproduce the simulation results of the FSW, OR, DR and FLR in Tan et al. (2025).
% Reference: Tan R., Huang W., Zhang Z. and Yin G. (2025). Causal effect of
% functional treatment. Journal of Machine Learning Research, 26, 1--39.
% Matlab version: R2023b.

n_sample = [200 500]; % Sample sizes
n_rep = 200; % Number of repetitions, 200 may take a long time to run.
t_num = 200;
t = linspace(0,1,t_num);
model_opt = 1; % Model choices, from 1 to 6, corresponding to the
% simulation models (i) to (vi) in the paper.

for n_sample_ind = 1:length(n_sample)
    n = n_sample(n_sample_ind);
 
    b_all_FSW = zeros(n_rep,t_num);
    b_all_OR = zeros(n_rep,t_num);
    b_X_all_OR = cell(n_rep,1);
    b_all_DR = zeros(n_rep,t_num); 
    b_all_naive = zeros(n_rep,t_num); % naive corresponds to the FLR in the paper,
    % same for what follows.
    
    a_all_FSW = zeros(n_rep,1);
    a_all_OR = zeros(n_rep,1);
    a_all_DR = zeros(n_rep,1); 
    a_all_naive = zeros(n_rep,1);
    
    d_opt_all = zeros(n_rep,1);
    n_h_opt_all = zeros(n_rep,1);
    k_2_opt_all = zeros(n_rep,1);
        
    MSE_ADRF_FSW = zeros(n_rep,1);
    MSE_ADRF_OR = zeros(n_rep,1);
    MSE_ADRF_DR = zeros(n_rep,1); 
    MSE_ADRF_naive = zeros(n_rep,1);

    parfor n_rep_ind = 1:n_rep
        rng(33*n_rep_ind)
        %% Generate data
        A = random('Normal',0,1,[n,6]).*[4 2*sqrt(3) 2*sqrt(2) 2 1 1/sqrt(2)];
        Z = A(:,1) * sqrt(2) * sin(2*pi*t) + A(:,2) * sqrt(2) * cos(2*pi*t) +...
            A(:,3) * sqrt(2) * sin(4*pi*t) + A(:,4) * sqrt(2) * cos(4*pi*t) +...
            A(:,5) * sqrt(2) * sin(6*pi*t) + A(:,6) * sqrt(2) * cos(6*pi*t);
        b = 2*sqrt(2)*sin(2*pi*t) + sqrt(2)*cos(2*pi*t) + sqrt(2)*sin(4*pi*t)/2 + sqrt(2)*cos(4*pi*t)/2;
        
        if model_opt == 1
        % Model (i)--------------------------------------------------------
        X = A(:,1)/4 + random('Normal',0,1,[n,1]);
        Y = 1 + trapz(t,Z.*b,2) + 2.*X + random('Normal',0,5,[n,1]); % mean(m(X)) = 0
        ADRF_true = 1 + trapz(t,Z.*b,2);

        elseif model_opt == 2
        % Model (ii)--------------------------------------------------------
        X = A(:,1)/4 + 0.25.*random('Normal',0,1,[n,1]);
        Y = 1 + trapz(t,Z.*b,2) + 5.*sin(X) + random('Normal',0,5,[n,1]); % mean(m(X)) = 0
        ADRF_true = 1  + trapz(t,Z.*b,2); 

        elseif model_opt == 3
        % Model (iii)-------------------------------------------------------- 
        X = zeros(n,2);
        X(:,1) = (A(:,1)/4 + 1).^2 + random('Normal',0,1,[n,1]);
        X(:,2) = A(:,2)/(4*sqrt(3)) ; 
        Y = 1 + trapz(t,Z.*b,2) + 2.*X(:,1) + 2.*X(:,2) + random('Normal',0,5,[n,1]); % mean(m(X)) = 4
        ADRF_true = 5 + trapz(t,Z.*b,2);

        elseif model_opt == 4
        % Model (iv)--------------------------------------------------------
        X = zeros(n,2);
        X(:,1) = (A(:,1)/4 + 1).^2 + random('Normal',0,1,[n,1]);
        X(:,2) = A(:,2)/(4*sqrt(3)) ; 
        Y = 1 + trapz(t,Z.*b,2) + 2.*X(:,1) + 2.*cos(X(:,1)) + 5.5.*sin(X(:,2)) + random('Normal',0,5,[n,1]); % mean(m(X)) = 4.396
        ADRF_true = 5.396 + trapz(t,Z.*b,2);

        elseif model_opt == 5
        % Model (v)--------------------------------------------------------
        X = zeros(n,2);
        X(:,1) = (A(:,1)/4 + 1).^2 + random('Normal',0,1,[n,1]);
        X(:,2) = A(:,2)/(4*sqrt(3)) ; 
        Y = 1 + trapz(t,Z.*b,2) + (trapz(t,Z.*b,2)).^2/25 + 2.*X(:,1) + 5.5.*sin(X(:,2)) + random('Normal',0,5,[n,1]); % mean(m(X)) = 4
        ADRF_true = 5 + trapz(t,Z.*b,2);
 
        elseif model_opt == 6 
        % Model (vi)--------------------------------------------------------
        X = random('Normal',0,1,[n,4]);
        A_1 = 4.*X(:,1) + random('Normal',0,1,[n,1]);
        A_2 = 2.*sqrt(3).*X(:,2) + random('Normal',0,1,[n,1]);
        A_3 = 2.*sqrt(2).*X(:,3) + random('Normal',0,1,[n,1]);
        A_4 = 2.*X(:,4) + random('Normal',0,1,[n,1]);
        Z = A_1.*sqrt(2).*sin(2.*pi.*t) + A_2.*sqrt(2).*sin(4.*pi.*t) ...
            + A_3.*sqrt(2).*sin(6.*pi.*t) + A_4.*sqrt(2).*sin(8.*pi.*t);
        Y = 10.*(X(:,2).*X(:,1).^2 + X(:,4).^2.*sin(2.*X(:,3))) + 0.5.*A_1.^2 +...
            4.*sin(A_1);
        ADRF_true = 0.5.*A_1.^2 + 4.*sin(A_1);
        end

        %% Perform different methods
        [xi,phi,lambda,mu_Z,~] = FPCA(t,Z,99);
        [~,~,~,~,d_95] = FPCA(t,Z,95);
        L = 10;
 
        if model_opt ~= 5 & model_opt ~= 6
            k_2_candi = 1:2;
            n_h_candi = round(n.*linspace(0.2,0.8,20));
        else
            k_2_candi = 2:3;
            n_h_candi = round(n.*linspace(0.6,0.8,20));
        end
 
        [~,d_opt] = CV_OR(t,Z,X,Y,L);
        [~,n_h_opt,k_2_opt] = CV_FSW_hk(t,Z,X,Y,L,d_opt,n_h_candi,k_2_candi);      
         
        d_opt_all(n_rep_ind) = d_opt;
        n_h_opt_all(n_rep_ind) = n_h_opt;
        k_2_opt_all(n_rep_ind) = k_2_opt;
        
        [pi_hat_NP,pi_hat_mat,Ker_Z] = weight_con_LOO(t,Z,X,n_h_opt,k_2_opt); 
                
        [a_hat_FSW,b_hat_FSW_coe,phi_hat] = FLR(t,Z,Y.*pi_hat_NP,d_opt);
        a_all_FSW(n_rep_ind,1) = a_hat_FSW;
        b_hat_FSW = sum(b_hat_FSW_coe.*phi_hat,1);
        b_all_FSW(n_rep_ind,:) = b_hat_FSW; 
        MSE_ADRF_FSW(n_rep_ind) = mean((a_hat_FSW + trapz(t,b_hat_FSW.*Z,2) - ADRF_true).^2);
        
        [a_hat_OR,b_hat_OR_coe,b_X,phi_hat] = FLR_mixed_BF(t,X,xi,lambda,phi,mu_Z,Y,d_opt);
        a_all_OR(n_rep_ind,1) = a_hat_OR;
        b_hat_OR = sum(b_hat_OR_coe.*phi_hat,1);
        b_all_OR(n_rep_ind,:) = b_hat_OR; 
        b_X_all_OR{n_rep_ind} = b_X; 
        MSE_ADRF_OR(n_rep_ind) = mean((a_hat_OR + trapz(t,b_hat_OR.*Z,2) + mean(X*b_X) - ADRF_true).^2);

        [a_hat_DR,b_hat_DR_coe,phi_hat] = FLR_DR(t,X,Z,Y,d_opt,n_h_opt,k_2_opt);
        a_all_DR(n_rep_ind,1) = a_hat_DR;
        b_hat_DR = sum(b_hat_DR_coe.*phi_hat,1);
        b_all_DR(n_rep_ind,:) = b_hat_DR;  
        MSE_ADRF_DR(n_rep_ind) = mean((a_hat_DR + trapz(t,b_hat_DR.*Z,2) - ADRF_true).^2);

        [a_hat_naive,b_hat_naive_coe,phi_hat_naive] = FLR(t,Z,Y,d_opt);
        a_all_naive(n_rep_ind,1) = a_hat_naive;
        b_hat_naive = sum(b_hat_naive_coe.*phi_hat_naive,1);
        b_all_naive(n_rep_ind,:) = b_hat_naive; 
        MSE_ADRF_naive(n_rep_ind) = mean((a_hat_naive + trapz(t,b_hat_naive.*Z,2) - ADRF_true).^2);
    
    end
    %% Save and print results 
    fname = sprintf('(%d)n%d_250514_results',model_opt,n); % Revise the file name as needed.
    save(fname,'b_X_all_OR','b_all_OR','b_all_FSW','b_all_DR','b_all_naive',...
        'a_all_OR','a_all_FSW','a_all_DR','a_all_naive',...
        'd_opt_all','n_h_opt_all','k_2_opt_all',...
        'MSE_ADRF_FSW','MSE_ADRF_OR','MSE_ADRF_DR','MSE_ADRF_naive');
    
    fprintf(fname)
    fprintf('\n')
    fprintf('Mean (standard deviation) of MSE: \n') 
    fprintf('FSW: %0.2f (%0.2f) \n',mean(MSE_ADRF_FSW),std(MSE_ADRF_FSW));
    fprintf('OR: %0.2f (%0.2f) \n',mean(MSE_ADRF_OR),std(MSE_ADRF_OR));
    fprintf('DR: %0.2f (%0.2f) \n',mean(MSE_ADRF_DR),std(MSE_ADRF_DR));
    fprintf('naive: %0.2f (%0.2f) \n',mean(MSE_ADRF_naive),std(MSE_ADRF_naive));
    fprintf('\n')
end
        
        
        
        
