function[betaout, lambdaout] = horseshoe(y, X, n_burnin, n_post_burnin, thin, scl_ub, scl_lb, n_warmup, a0, b0, beta_true)
% Function to impelement Horseshoe shrinkage prior (http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Horse.pdf)
% in Bayesian Linear Regression. %%
% Based on code by Antik Chakraborty (antik@stat.tamu.edu) and Anirban Bhattacharya (anirbanb@stat.tamu.edu)
% Modified by James Johndrow (johndrow@stanford.edu)
% Modified further by Akihiko Nishimura

% Model: y = X\beta + \epslion, \epsilon \sim N(0, \sigma^2) %%
%        \beta_j \sim N(0, \sigma^2 \lambda_j^2 \tau^2) %%
%        \lambda_j \sim Half-Cauchy(0, 1), \tau \sim Half-Cauchy (0, 1) %%
%        \pi(\sigma^2) \sim 1/\sigma^2 %%


% This function implements the algorithm proposed in "Scalable MCMC for
% Bayes Shrinkage Priors" by Johndrow and Orenstein (2017).
% The global local scale parameters are updated via a Slice sampling scheme given in the online supplement
% of "The Bayesian Bridge" by Polson et. al. (2011). Setting ab = true
% implements the algorithm of Bhattacharya et al. Setting ab = false
% implements the algorith of Johndrow and Orenstein, which uses a block
% update for \tau, \sigma^2, \beta



% Input: y = response, a n * 1 vector %%
%        X = matrix of covariates, dimension n * p %%
%        BURNIN= number of burnin MCMC samples %%
%        MCMC= number of posterior draws to be saved %%
%        thin= thinning parameter of the chain %%
%        scl_ub = upper bound on scale for MH proposals
%        scl_lb = lower bound on scale for MH proposals (usually make these
%        equal; 0.8 a good default)
%        n_warmup = number of iterations over which to transition between upper
%        and lower bound on MH proposals; usually make this 1 and just make
%        scl_ub = scl_lb; only use for particularly challenging cases
%        SAVE_SAMPLES = binary; whether to save samples
%        ab = whether to run the algorithm of Bhattachaya et al.
%        trunc = whether to use the numeric truncations of Bhattacharya et al
%        a0 = parameter of gamma prior for sigma2
%        b0 = second parameter of gamma prior for sigma2
%        BetaTrue = true beta (for simulations)
%        disp_int = how often to produce output
%        plotting = whether to make plots
%        corX = whether simulations were performed with correlated design



% Output:
%         pMean= posterior mean of Beta, a p by 1 vector%%
%         pMeadian = posterior median of Beta, a p by 1 vector %%
%         pLambda = posterior mean of local scale parameters, a p by 1 vector %%
%         pSigma = posterior mean of Error variance %%
%         betaout = posterior samples of beta %%


tic;
n_iter = n_burnin + n_post_burnin;
n_sample = ceil(n_post_burnin / thin); % Number of samples to keep
[n, p] = size(X);

% paramters %
beta = zeros(p, 1);
lambda = ones(p, 1);
tau = 1;
sigma_sq = 1;

% output %
betaout = zeros(50, n_sample);
lambdaout = zeros(50, n_sample);
etaout = zeros(50, n_sample);
tauout = zeros(n_sample, 1);
xiout = zeros(n_post_burnin + n_burnin, 1);
sigmaSqout = zeros(n_sample, 1);
l1out = zeros(n_post_burnin + n_burnin, 1);
pexpout = zeros(n_post_burnin + n_burnin, 1);
ACC = zeros(n_post_burnin + n_burnin, 1);

% matrices %
I_n = eye(n);
l = ones(n, 1);
xi = tau^(-2);

% start Gibbs sampling %
for i = 1:n_iter

    % update tau %
    LX = bsxfun(@times, (lambda.^2), X');
    XLX = X * LX;

    eta = lambda.^(-2);
    if i<n_warmup
       std_MH = (scl_ub .* (n_warmup - i) + scl_lb .* i) ./ n_warmup;
    else
       std_MH = scl_lb;
    end
    prop_xi = exp(normrnd(log(xi), std_MH));
    [lrat_prop, M_chol_prop] = lmh_ratio(XLX, y, prop_xi, I_n, n, a0, b0);
    [lrat_curr, M_chol_curr] = lmh_ratio(XLX, y, xi, I_n, n, a0, b0);
    log_acc_rat = (lrat_prop - lrat_curr) + (log(prop_xi) - log(xi));

    ACC(i) = (rand < exp(log_acc_rat));
    if ACC(i) % if accepted, update
        xi = prop_xi;
        M_chol = M_chol_prop;
    else
        M_chol = M_chol_curr;
    end
    tau = 1 ./ sqrt(xi);

    % update sigma_sq marginal of beta %
    xtmp = cho_solve(M_chol, y);
    ssr = y' * xtmp;
    sigma_sq = 1 / gamrnd((n + a0) / 2, 2 / (ssr + b0));

    % Alternative update of sigma_sq conditional on beta
    %{
        if trunc
            E_1 = max((y - X * Beta)' * (y - X * Beta), (1e-10)); % for numerical stability
            E_2 = max(sum(Beta.^2 ./ ((tau * lambda)).^2), (1e-10));
        else
            E_1 = (y - X * Beta)' * (y - X * Beta); E_2 = sum(Beta.^2 ./ ((tau * lambda)).^2);
        end

        sigma_sq = 1 / gamrnd((n + p + a0) / 2, 2 / (b0 + E_1 + E_2));
    %}

    % update beta %
    U = (1 ./ xi) .* LX;
    u = normrnd(0, tau * lambda);
    v = X * u + normrnd(0, l);
    v_star= cho_solve(M_chol, (y ./ sqrt(sigma_sq)) - v);
    beta = sqrt(sigma_sq) * (u + U * v_star);

    % update lambda_j's in a block using slice sampling %
    u = unifrnd(0, 1 ./ (eta + 1));
    gamma_rate = (beta.^2) .* xi ./ (2 .* sigma_sq);
    eta = gen_truncated_exp(gamma_rate, (1 - u) ./ u);
    if any(eta <= 0)
       disp([num2str(sum(eta <= 0)) ' Eta underflowed, replacing = machine epsilon']);
       eta(eta <= 0) = eps;
    end
    lambda = 1 ./ sqrt(eta);

    % (theoretically) equivalent way to sample lambda_j's, but supposedly
    % not as numerically stable.
    %{
    eta = 1 ./ (lambda.^2);
        upsi = unifrnd(0, 1 ./ (1 + eta));
        tempps = Beta.^2 / (2 * sigma_sq * tau^2);
        ub = (1 - upsi) ./ upsi;

        % now sample eta from exp(tempv) truncated between 0 & upsi / (1 - upsi)
        Fub = 1 - exp( - tempps .* ub); % exp cdf at ub
        Fub(Fub < (1e-4)) = 1e-4;  % for numerical stability
        up = unifrnd(0, Fub);
        eta = -log(1 - up) ./ tempps;
        lambda = 1 ./ sqrt(eta);
        Eta = eta;
    %}

    per_expl = 1 - sqrt(sum((beta - beta_true).^2)) ./ sqrt(sum(beta_true.^2));
    L1_loss = 1 - sum(abs(beta - beta_true)) ./ sum(abs(beta_true));

    if i > n_burnin && mod(i, thin) == 0
        betaout(:, (i - n_burnin) / thin) = beta(1:50);
        lambdaout(:, (i - n_burnin) / thin) = lambda(1:50);
        etaout(:, (i - n_burnin) / thin) = eta(1:50);
        tauout((i - n_burnin) / thin) = tau;
        xiout(i) = xi;
        sigmaSqout((i - n_burnin) / thin) = sigma_sq;
        l1out(i) = L1_loss;
        pexpout(i) = per_expl;
    end
end

t = toc;
fprintf('Execution time of %d Gibbs iteration with (n, p) = (%d, %d)is %f seconds. \n', n_iter, n, p, t)

end



function [lr, M_chol] = lmh_ratio(XLX, y, xi, I_n, n, a0, b0)
    % marginal of beta, sigma2
    try
        M = I_n + (1 ./ xi) .* XLX;
        M_chol = chol(M);
        x = cho_solve(M_chol, y);
        ssr = y' * x + b0;
        ldetM = 2 * sum(log(diag(M_chol)));
        ll = - .5 .* ldetM - ((n + a0) / 2) .* log(ssr);
        lpr = - log(sqrt(xi) .* (1 + xi));
        lr = ll + lpr;
    catch
        lr = - Inf; warning('proposal was rejected because I + XDX was not positive-definite');
    end
end

function x = cho_solve(R, b)
%%% Solve the system R * R' * x = b where R is upper triangular.
    x = linsolve(R, linsolve(R', b, struct('LT', true)), struct('UT', true));
end

function x = gen_truncated_exp(mn, trunc_point)
    r = mn .* trunc_point;
    sml = abs(r)<eps;
    x = zeros(max(length(mn), length(trunc_point)), 1);
    tmp = zeros(max(length(mn), length(trunc_point)), 1);

    if any(sml)
        tmp(sml) = expm1( - r(sml)) .* rand(length(mn(sml)), 1);
    end
    tmp(~sml) = (exp( - r(~sml)) - 1) .* rand(length(mn(~sml)), 1);

    sml = abs(tmp) < eps;
    if any(sml)
       x(sml) = - log1p(tmp(sml)) ./ mn(sml);
    end
    x(~sml) = - log(1 + tmp(~sml)) ./ mn(~sml);

end
