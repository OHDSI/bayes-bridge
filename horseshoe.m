function[pMean,pMedian,pLambda,pSigma,betaout]=horseshoe(y,X,BURNIN,MCMC,thin,scl_ub,scl_lb,phasein,SAVE_SAMPLES,ab,trunc,a0,b0,BetaTrue,disp_int,plotting,corX)
% Function to impelement Horseshoe shrinkage prior (http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Horse.pdf)
% in Bayesian Linear Regression. %%
% Based on code by Antik Chakraborty (antik@stat.tamu.edu) and Anirban Bhattacharya (anirbanb@stat.tamu.edu)
% Modified by James Johndrow (johndrow@stanford.edu)

% Model: y=X\beta+\epslion, \epsilon \sim N(0,\sigma^2) %%
%        \beta_j \sim N(0,\sigma^2 \lambda_j^2 \tau^2) %%
%        \lambda_j \sim Half-Cauchy(0,1), \tau \sim Half-Cauchy (0,1) %%
%        \pi(\sigma^2) \sim 1/\sigma^2 %%


% This function implements the algorithm proposed in "Scalable MCMC for
% Bayes Shrinkage Priors" by Johndrow and Orenstein (2017). 
% The global local scale parameters are updated via a Slice sampling scheme given in the online supplement 
% of "The Bayesian Bridge" by Polson et. al. (2011). Setting ab = true
% implements the algorithm of Bhattacharya et al. Setting ab=false
% implements the algorith of Johndrow and Orenstein, which uses a block
% update for \tau, \sigma^2, \beta



% Input: y=response, a n*1 vector %%
%        X=matrix of covariates, dimension n*p %%
%        BURNIN= number of burnin MCMC samples %%
%        MCMC= number of posterior draws to be saved %%
%        thin= thinning parameter of the chain %%
%        scl_ub=upper bound on scale for MH proposals
%        scl_lb=lower bound on scale for MH proposals (usually make these
%        equal; 0.8 a good default)
%        phasin=number of iterations over which to transition between upper
%        and lower bound on MH proposals; usually make this 1 and just make
%        scl_ub=scl_lb; only use for particularly challenging cases
%        SAVE_SAMPLES=binary; whether to save samples
%        ab=whether to run the algorithm of Bhattachaya et al.
%        trunc=whether to use the numeric truncations of Bhattacharya et al
%        a0=parameter of gamma prior for sigma2
%        b0=second parameter of gamma prior for sigma2
%        BetaTrue=true beta (for simulations)
%        disp_int=how often to produce output
%        plotting=whether to make plots
%        corX=whether simulations were performed with correlated design



% Output: 
%         pMean= posterior mean of Beta, a p by 1 vector%%
%         pMeadian=posterior median of Beta, a p by 1 vector %%
%         pLambda=posterior mean of local scale parameters, a p by 1 vector %%
%         pSigma=posterior mean of Error variance %% 
%         betaout=posterior samples of beta %%



%profile on;


tic;
N=BURNIN+MCMC;
effsamp=(N-BURNIN)/thin;
[n,p]=size(X);

% paramters %
Beta=ones(p,1); 
lambda=ones(p,1);
tau=1; sigma_sq=1;

% output %
betaout=zeros(50,effsamp);
lambdaout=zeros(50,effsamp);
etaout = zeros(50,effsamp);
tauout=zeros(effsamp,1);
xiout = zeros(MCMC+BURNIN,1);
sigmaSqout=zeros(effsamp,1);
l1out = zeros(MCMC+BURNIN,1);
pexpout = zeros(MCMC+BURNIN,1);
ACC = zeros(MCMC+BURNIN,1);

% matrices %
I_n=eye(n); 
l=ones(n,1);

Xi = tau^(-2);
Eta = lambda.^(-2);

% start Gibbs sampling %
for i=1:N  
    
    % update tau %
    if i>0
        if ab
            tempt = sum((Beta./lambda).^2)/(2*sigma_sq);
            et = 1/tau^2;
            utau = unifrnd(0,1/(1+et));
            ubt = (1-utau)/utau;
            Fubt = gamcdf(ubt,(p+1)/2,1/tempt);
            if trunc
                Fubt = max(Fubt,1e-8); % for numerical stability
            end
            ut = unifrnd(0,Fubt);
            et = gaminv(ut,(p+1)/2,1/tempt);
            tau = 1/sqrt(et);
            Xi = et;
        else
            LX=bsxfun(@times,(lambda.^2),X');
            XLX = X*LX;
            
            Eta = lambda.^(-2);
            if i<phasein
               std_MH = (scl_ub.*(phasein-i)+scl_lb.*i)./phasein; 
            else
               std_MH = scl_lb; 
            end
            prop_Xi = exp(normrnd(log(Xi),std_MH));
            lrat_prop = lmh_ratio(XLX,y,prop_Xi,I_n,n,a0,b0);
            lrat_curr = lmh_ratio(XLX,y,Xi,I_n,n,a0,b0);
            log_acc_rat = (lrat_prop-lrat_curr)+(log(prop_Xi)-log(Xi));
            
            ACC(i) = (rand < exp(log_acc_rat));
            if ACC(i) % if accepted, update
                Xi = prop_Xi;
            end
            tau = 1./sqrt(Xi);
        end
    end
    
    % update sigma_sq %
    if ab
        % new method
        if trunc
            E_1=max((y-X*Beta)'*(y-X*Beta),(1e-10)); % for numerical stability
            E_2=max(sum(Beta.^2./((tau*lambda)).^2),(1e-10));
        else
            E_1 = (y-X*Beta)'*(y-X*Beta); E_2 = sum(Beta.^2./((tau*lambda)).^2);
        end
        
        sigma_sq=1/gamrnd((n+p+a0)/2,2/(b0+E_1+E_2));
        
    else
        % update sigma2 marginal of beta %
        M = I_n + (1./Xi).*XLX;
        xtmp = M\y;
        ssr = y'*xtmp;
        
        sigma_sq = 1/gamrnd((n+a0)/2,2/(ssr+b0));
    end

    % update beta %
    if ab
        lambda_star=tau*lambda;
        U=bsxfun(@times,(lambda_star.^2),X');
        M = I_n + X*U;
    else
        U = (1./Xi).*LX;
    end

    % step 1 %
    u=normrnd(0,tau*lambda);
    v=X*u+normrnd(0,l);

    % step 2 %
    v_star=(M)\((y./sqrt(sigma_sq))-v);
    Beta=sqrt(sigma_sq)*(u+U*v_star);
    


    
    % update lambda_j's in a block using slice sampling %
    if ab && trunc
        eta = 1./(lambda.^2); 
        upsi = unifrnd(0,1./(1+eta));
        tempps = Beta.^2/(2*sigma_sq*tau^2); 
        ub = (1-upsi)./upsi;

        % now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
        Fub = 1 - exp(-tempps.*ub); % exp cdf at ub 
        Fub(Fub < (1e-4)) = 1e-4;  % for numerical stability
        up = unifrnd(0,Fub); 
        eta = -log(1-up)./tempps; 
        lambda = 1./sqrt(eta);
        Eta = eta;
    else
        u = unifrnd(0, 1./(Eta+1));
        gamma_rate = (Beta.^2) .* Xi ./ (2.*sigma_sq);
        Eta = gen_truncated_exp(gamma_rate, (1-u)./u);
        if any(Eta<=0)
           disp([num2str(sum(Eta<=0)) ' Eta underflowed, replacing = machine epsilon']);
           Eta(Eta<=0) = eps;
        end
        %Eta(Eta<=0) = 1e-10;
        lambda = 1./sqrt(Eta);
     end

    if mod(i,disp_int) == 0
        %toc
        %tic;
        disp(i);
        disp(mean(ACC(1:i)));
        Beta(1:20)
        if plotting
            figure(1);subplot(2,2,1);plot(log(xiout(50:(i-1))),'.');title('log(\xi)');
            subplot(2,2,2);plot(sigmaSqout(50:(i-1)),'.');title('\sigma^2');
            subplot(2,2,3);plot(betaout(1,50:(i-1)),'.');title('\beta_1');
            subplot(2,2,4);plot(etaout(1,50:(i-1)),'.');title('\eta_1');
            drawnow;
            figure(2); subplot(1,2,1);plot(l1out(1:(i-1)),'.');title('L1');ylim([0 1]);
            subplot(1,2,2);plot(pexpout(1:(i-1)),'.');title('pexpl'); ylim([0 1]);
            drawnow;
        end
    end
    
    per_expl = 1-sqrt(sum((Beta-BetaTrue).^2))./sqrt(sum(BetaTrue.^2));
    L1_loss = 1-sum(abs(Beta-BetaTrue))./sum(abs(BetaTrue));
    
    if i > BURNIN && mod(i, thin)== 0
        betaout(:,(i-BURNIN)/thin) = Beta(1:50);
        lambdaout(:,(i-BURNIN)/thin) = lambda(1:50);
        etaout(:,(i-BURNIN)/thin) = Eta(1:50);
        tauout((i-BURNIN)/thin)=tau;
        xiout(i) = Xi;
        sigmaSqout((i-BURNIN)/thin)=sigma_sq;
        l1out(i) = L1_loss;
        pexpout(i) = per_expl;
    end
end
pMean=mean(betaout,2);
pMedian=median(betaout,2);
pSigma=mean(sigmaSqout);
pLambda=mean(lambdaout,2);
t=toc;
fprintf('Execution time of %d Gibbs iteration with (n,p)=(%d,%d)is %f seconds',N,n,p,t)
if SAVE_SAMPLES
    if ~ab
        if corX
            save(strcat('Outputs/post_reg_horse_corX_',num2str(n),'_',num2str(p),'.mat'),'betaout','lambdaout','etaout','tauout','xiout','sigmaSqout','l1out','pexpout','t');  
        else
            save(strcat('Outputs/post_reg_horse_',num2str(n),'_',num2str(p),'.mat'),'betaout','lambdaout','etaout','tauout','xiout','sigmaSqout','l1out','pexpout','t');  
        end
    else
        if trunc
            if corX
                save(strcat('Outputs/post_reg_horse_ab_trunc_corX_',num2str(n),'_',num2str(p),'.mat'),'betaout','lambdaout','etaout','tauout','xiout','sigmaSqout','l1out','pexpout','t');
            else
                save(strcat('Outputs/post_reg_horse_ab_trunc_',num2str(n),'_',num2str(p),'.mat'),'betaout','lambdaout','etaout','tauout','xiout','sigmaSqout','l1out','pexpout','t');
            end
        else
            if corX
                save(strcat('Outputs/post_reg_horse_ab_corX_',num2str(n),'_',num2str(p),'.mat'),'betaout','lambdaout','etaout','tauout','xiout','sigmaSqout','l1out','pexpout','t');
            else
                save(strcat('Outputs/post_reg_horse_ab_',num2str(n),'_',num2str(p),'.mat'),'betaout','lambdaout','etaout','tauout','xiout','sigmaSqout','l1out','pexpout','t');
            end
        end
    end
end

%profile viewer;
end



function lr = lmh_ratio(XLX,y,Xi,I_n,n,a0,b0)
    % marginal of beta, sigma2
    M = I_n + (1./Xi).*XLX;
    x = M\y;
    ssr = y'*x+b0;
    try
        cM = chol(M); 
        ldetM = 2*sum(log(diag(cM)));
        ll = -.5.*ldetM - ((n+a0)/2).*log(ssr);
        lpr = -log(sqrt(Xi).*(1+Xi));
        lr = ll+lpr;
    catch
        lr = -Inf; warning('proposal was rejected because I+XDX was not positive-definite');
    end
end

function x = gen_truncated_exp(mn,trunc_point)
    r = mn.*trunc_point;
    sml = abs(r)<eps;
    x = zeros(max(length(mn),length(trunc_point)),1);
    tmp = zeros(max(length(mn),length(trunc_point)),1);
    
    if any(sml)
        tmp(sml) = expm1(-r(sml)).*rand(length(mn(sml)),1);
    end
    tmp(~sml) = (exp(-r(~sml))-1).*rand(length(mn(~sml)),1);
    
    sml = abs(tmp)<eps;
    if any(sml)
       x(sml) = -log1p(tmp(sml))./mn(sml); 
    end
    x(~sml) = -log(1+tmp(~sml))./mn(~sml);
    
end



