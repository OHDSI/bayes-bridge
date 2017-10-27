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
    SigmaTrue = 2;
    BetaTrue = zeros(p, 1);
    BetaTrue(1:5) = 4;
    BetaTrue(6:15) = 2.^(-(0:.5:4.5));

    % Basic Variables
    if corX
        X = normrnd(0, 1, [n p]);
        for j = 2:p
            X(:, j) = rhoX.*X(:, j-1) + X(:, j);
        end
    else
        X = normrnd(0, 1, [n p]);
    end
    y = X*BetaTrue + SigmaTrue.*normrnd(0, 1, [n 1]);

    scl_ub = 3; % scale for Metropolis-Hastings proposals for xi
    scl_lb = .8; % lb scale
    phasein = 1;
    nmc = 100; % length of Markov chain
    burn = 0; % number of burn-ins
    a0 = 1/2; b0 = 1/2; % Hyper-params on the prior for sigma_sq. Jeffrey's prior would be a0 = b0 = 0.

    % Running horse_nmean_mh
    BURNIN = 0; MCMC = nmc; thin = 1; plotting = true;
    profile on
    [betaout_new, lambda_out] = horseshoe(y, X, BURNIN, MCMC, thin, scl_ub, scl_lb, phasein, a0, b0, BetaTrue);
    profile off
end

% Make sure that the output of the code is identical to the one before
% modifications were made.
tol = 10^-6;
load('output.mat')
if all(all(abs(betaout - betaout_new) < tol))
    disp('The current output agrees with the previous one.')
else
    disp('WARNING! The current output does NOT agree with the previous error.')
    disp('Some bugs have likely been introduced to the code.')
end
