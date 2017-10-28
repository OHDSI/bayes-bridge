clear;
corX = false;
rhoX = .9;
%ps = [2000 3000 4000 5000 6000 7000 8000 9000 10000 15000 20000];
ps = 1000;
n = 1000; % number of sample points
for p = ps
    disp(num2str(p));
    rng(571);
    %p = 10000; % number of parameters

    % True parameters
    sigma_true = 2;
    beta_true = zeros(p, 1);
    beta_true(1:5) = 4;
    beta_true(6:15) = 2.^(-(0:.5:4.5));

    % Basic Variables
    if corX
        X = normrnd(0, 1, [n p]);
        for j = 2:p
            X(:, j) = rhoX .* X(:, j-1) + X(:, j);
        end
    else
        X = normrnd(0, 1, [n p]);
    end
    y = X * beta_true + sigma_true.*normrnd(0, 1, [n 1]);

    scl_ub = 3; % scale for Metropolis-Hastings proposals for xi
    scl_lb = .8; % lb scale
    phasein = 1;
    nmc = 100; % length of Markov chain
    burn = 0; % number of burn-ins

    % Running horse_nmean_mh
    n_burnin = 0; n_post_burnin = nmc; thin = 1; plotting = true;
    fix_tau = false;
    tau = 10^-3;
    profile on
    [beta_samples, lambda_samples, tau_samples] = gibbs(y, X, n_burnin, n_post_burnin, thin, scl_ub, scl_lb, phasein, fix_tau, tau);
    profile off
end

% Make sure that the output of the code is identical to the one before
% modifications were made.
tol = 10^-6;
load('output.mat')
if all(all(abs(betaout - beta_samples) < tol))
    disp('The current output agrees with the previous one.')
else
    disp('WARNING! The current output does NOT agree with the previous error.')
    disp('Some bugs have likely been introduced to the code.')
end

%%
subplot(2, 1, 1)
plot(beta_samples')
subplot(2, 1, 2)
plot(lambda_samples')