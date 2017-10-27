clear; 
ab = false;trunc = false;SAVE_SAMPLES=true;disp_int = 2500; corX = false; rhoX = .9;
%ps = [2000 3000 4000 5000 6000 7000 8000 9000 10000 15000 20000];
ps = 1000;
n = 1000; % number of sample points
for p=ps
    disp(num2str(p));
    rng(571);
    sim_type = 'F'; % Frequentist or Bayesian
    %p = 10000; % number of parameters

    % True parameters
    TauTrue = .01;
    if strcmp(sim_type, 'B') % 'Bayesian': BetaTrue is coming from a distribution
      SigmaTrue = 2;
      LambdaTrue = trnd(1,[p 1]);
      BetaTrue = normrnd(0,SigmaTrue.*TauTrue.*LambdaTrue);
    else % 'Frequentist': BetaTrue is an unknown, deterministic, vector
      SigmaTrue = 2;
      BetaTrue = zeros(p,1);
      BetaTrue(1:5) = 4;
      BetaTrue(6:15) = 2.^(-(0:.5:4.5));
    end

    % Basic Variables
    if corX
        X = normrnd(0,1,[n p]);
        for j=2:p
            X(:,j) = rhoX.*X(:,j-1)+X(:,j);
        end
    else
        X = normrnd(0,1,[n p]);
    end
    y = X*BetaTrue+SigmaTrue.*normrnd(0,1,[n 1]);
    %X = sparse(X);

    Sigma2Est = 1.0; % type: Float64
    TauEst = TauTrue;
    LambdaEst = ones(p);
    %LambdaEst = LambdaTrue;
    scl_ub = 3; % scale for Metropolis-Hastings proposals for xi
    scl_lb = .8; % lb scale 
    phasein = 1;
    slice_lambda = true; % whether to update lambda via slice sampling
    nmc = 40000; % length of Markov chain
    burn = 0; % number of burn-ins
    % disp_int = 1000; % display interval; outputs parameter estimates while running
    % plotting = false # whether to plot diagnostics while running
    a0 = 1/2; b0 = 1/2;

    % Running horse_nmean_mh
    BURNIN = 0; MCMC = nmc; thin = 1; plotting = true; 
    [pMean,pMedian,pLambda,pSigma,betaout]=horseshoe(y,X,BURNIN,MCMC,thin,scl_ub,scl_lb,phasein,SAVE_SAMPLES,ab,trunc,a0,b0,BetaTrue,disp_int,plotting,corX);
end








