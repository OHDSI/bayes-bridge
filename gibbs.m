function[beta_samples, lambda_samples, tau_samples] = ...
    gibbs(y, X, n_burnin, n_post_burnin, thin, fixed_tau, tau, lambda0)
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
%        n_burnin = number of burnin MCMC samples %%
%        n_post_burnin = number of posterior draws to be saved %%
%        thin = thinning parameter of the chain %%
%        and lower bound on MH proposals; usually make this 1 and just make
%        scl_ub = scl_lb; only use for particularly challenging cases
%        fixed_tau = if true, tau will not be updated.
%        lambda0 = the initial value for MCMC

tic;
n_iter = n_burnin + n_post_burnin;
n_sample = ceil(n_post_burnin / thin); % Number of samples to keep
[n, p] = size(X);

% Hyper-params on the prior for sigma_sq. Jeffrey's prior would be a0 = b0 = 0.
a0 = .5;
b0 = .5;

% Stepsize of Metropolis. Apparently, .8 is a good default.
std_MH = .8;

% paramters %
beta = zeros(p, 1); % Unused with the current gibbs update order.
sigma_sq = 1; % Unused with the current gibbs update order.
if ~isnan(lambda0)
    lambda = lambda0;
else
    lambda = ones(p, 1);
end
if ~fixed_tau
    tau = 1;
end


% output %
beta_samples = zeros(p, n_sample);
lambda_samples = zeros(p, n_sample);
tau_samples = zeros(n_sample, 1);
sigmaSq_samples = zeros(n_sample, 1);
accept_prob = zeros(n_post_burnin + n_burnin, 1);

% matrices %
I_n = eye(n);
l = ones(n, 1);
xi = tau^(-2);
eta = lambda.^(-2);

% start Gibbs sampling %
for i = 1:n_iter
    
    LX = bsxfun(@times, (lambda.^2), X');
    XLX = X * LX;
    
    % update tau %
    if fixed_tau
        M = I_n + (1 ./ xi) .* XLX;
        M_chol = chol(M);
    else
        prop_xi = exp(normrnd(log(xi), std_MH));
        [lrat_prop, M_chol_prop] = lmh_ratio(XLX, y, prop_xi, I_n, n, a0, b0);
        [lrat_curr, M_chol_curr] = lmh_ratio(XLX, y, xi, I_n, n, a0, b0);
        log_acc_rat = (lrat_prop - lrat_curr) + (log(prop_xi) - log(xi));

        accept_prob(i) = (rand < exp(log_acc_rat));
        if accept_prob(i) % if accepted, update
            xi = prop_xi;
            M_chol = M_chol_prop;
        else
            M_chol = M_chol_curr;
        end
        tau = 1 ./ sqrt(xi);
    end

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

    if i > n_burnin && mod(i, thin) == 0
        beta_samples(:, (i - n_burnin) / thin) = beta;
        lambda_samples(:, (i - n_burnin) / thin) = lambda;
        tau_samples((i - n_burnin) / thin) = tau;
        sigmaSq_samples((i - n_burnin) / thin) = sigma_sq;
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
