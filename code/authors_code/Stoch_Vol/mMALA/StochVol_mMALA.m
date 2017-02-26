function [] = StochVol_mMALA

% Random Numbers...
randn('state', sum(100*clock));
rand('twister', sum(100*clock));

% True Parameters
beta  = 0.65;
sigma = 0.15;
phi   = 0.98;


%%%%% Generate Data %%%%%
%{
NumOfObs        = 2000;

x = zeros(NumOfObs,1); % Latent Volatilities
y = zeros(NumOfObs,1); % Observed Data


% Generate latent volatility at t = 1
x(1) = normrnd(0, sigma/sqrt(1 - phi^2));

% Generate latent volatilities
for n = 1:(NumOfObs-1)
    x(n + 1) = phi*x(n) + normrnd(0, sigma);
end

% Generate observed data
y = beta*randn(NumOfObs,1).*exp(x/2);

Truex = x; % Save true x values
%}


%%%%% Load Saved Data %%%%%
%
Data  = load('StochVolData1.mat');
x     = Data.Truex;
Truex = Data.Truex;
y     = Data.y;
NumOfObs = length(y);
%}


% Sparse diagonal matrix
I = sparse(eye(NumOfObs));

% Preallocate off diagonal elements in sparse matrix for speed
ITri = I;
ITri(2:NumOfObs+1:end)          = 0.1; % Subdiagonal
ITri(NumOfObs+1:NumOfObs+1:end) = 0.1; % Superdiagonal


subplot(121);
plot(y)
subplot(122)
plot(x)

% Precompute for speed
ySquared    = y.^2;



% Setup Natural Langevin for Xs
NumOfIterations    = 30000;
BurnIn             = 10000;

StepSize           = 0.07;

% Setup Natural Langevin for HP
HPStepSize           = 1;

Proposed = 0;
Accepted = 0;

HPProposed = 0;
HPAccepted = 0;


% Set initial values of x, beta, sigma, phi
x     = y;
beta  = 0.5;
sigma = 0.5;
phi   = 0.5;

% Preallocate space for saving results
xSaved     = zeros(NumOfIterations-BurnIn,NumOfObs);
betaSaved  = zeros(NumOfIterations-BurnIn,1);
phiSaved   = zeros(NumOfIterations-BurnIn,1);
sigmaSaved = zeros(NumOfIterations-BurnIn,1);
LJLSaved   = zeros(NumOfIterations-BurnIn,1);



for IterationNum = 1:NumOfIterations

        
    if mod(IterationNum,20) == 0
        disp([num2str(IterationNum) ' iterations completed.'])
        drawnow
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample latent variables %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Update the log-jointlikelihood
    CurrentLJL =  -(1/(2*sigma^2))*x(1)^2*(1-phi^2) - sum( x./2 + (ySquared)./(2*beta^2*exp(x)) ) - (1/(2*sigma^2))*sum( (x(2:end)-phi*x(1:end-1)).^2 );
    
    xNew = x;
    Proposed = Proposed + 1;
    
    % Create Gradient & Metric Tensor
    s   = -0.5 + (ySquared)./(2*beta^2*exp(xNew));
    d_1 = (1/sigma^2)*(xNew(1) - phi*xNew(2));
    d_2 = (1/sigma^2)*(xNew(end) - phi*xNew(end-1));
    r   = [d_1; -(phi/sigma^2)*(xNew(3:end) - phi*xNew(2:end-1)) + (1/sigma^2)*(xNew(2:end-1) - phi*xNew(1:end-2)); d_2];

    % Gradient
    GradL = s - r;
    
    iC                            = ITri;
    iC(1:NumOfObs+1:end)          = (1+phi^2)/sigma^2;
    iC(1,1)                       = 1/sigma^2;
    iC(end,end)                   = 1/sigma^2;
    iC(2:NumOfObs+1:end)          = -phi/sigma^2; % Lower off diagonal
    iC(NumOfObs+1:NumOfObs+1:end) = -phi/sigma^2; % Upper off diagonal
    
    G                   = iC;
    G(1:NumOfObs+1:end) = G(1:NumOfObs+1:end) + 0.5; % Add diag of 0.5 i.e. Fisher information
    CholG               = chol(G);
    LogDetG             = 2*sum(log(diag(CholG)));
    
    
    % Update parameters
    % Constant metric tensor
    Mean = xNew + (StepSize/(2))*(G\GradL);
    
    R = randn(1,NumOfObs);
    xNew = Mean + ( (R*(CholG\eye(NumOfObs))).*(StepSize^0.5) )';
    
    % Calculate prob of New given Old
    ProbNewGivenOld = -0.5*( log(StepSize)-LogDetG ) - (0.5/StepSize)*((Mean-xNew)'*G*(Mean-xNew));
    
    
    % Create Gradient only - Metric Tensor doesn't change from above
    s   = -0.5 + (ySquared)./(2*beta^2*exp(xNew));
    d_1 = (1/sigma^2)*(xNew(1) - phi*xNew(2));
    d_2 = (1/sigma^2)*(xNew(end) - phi*xNew(end-1));
    r   = [d_1; -(phi/sigma^2)*(xNew(3:end) - phi*xNew(2:end-1)) + (1/sigma^2)*(xNew(2:end-1) - phi*xNew(1:end-2)); d_2];

    % Gradient
    GradL = s - r;
    
    % Constant metric tensor
    Mean = xNew + (StepSize/(2))*(G\GradL);
    
    % Calculate prob of New given Old
    ProbOldGivenNew = -0.5*( log(StepSize)-LogDetG ) - (0.5/StepSize)*((Mean-x)'*G*(Mean-x));
    
    
    % Calculate proposed log-joint-likelihood value
    ProposedLJL =  -(1/(2*sigma^2))*xNew(1)^2*(1-phi^2) - sum( xNew./2 + (ySquared)./(2*beta^2*exp(xNew)) ) - (1/(2*sigma^2))*sum( (xNew(2:end)-phi*xNew(1:end-1)).^2 );
    
    
    % Accept according to ratio
    Ratio = ProposedLJL + ProbOldGivenNew - CurrentLJL - ProbNewGivenOld;

    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        x = xNew;
        Accepted = Accepted + 1;
        %disp('Accepted')
        drawnow
    else
        %disp('Rejected')
        drawnow
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%
    % Sample parameters %
    %%%%%%%%%%%%%%%%%%%%%
    
    % Precalculate exp(x) for speed
    Expx = exp(x);
    
    CurrentLJL = -sum(x/2) - NumOfObs*log(beta) - sum( (ySquared)./(2*beta^2*Expx) ) + 0.5*log(1-phi^2) - log(sigma) - x(1)^2*(1-phi^2)/(2*sigma^2) - (NumOfObs-1)*log(sigma) - sum( (x(2:end)-phi*x(1:end-1)).^2./(2*sigma^2) );
    CurrentJ   = log(sigma*(1-phi^2)); % Jacobian
    % Add Prior
    Prior      = -beta - 0.5/(2*sigma^2) - 6*log(sigma^2) + log(sigma) + 19*log((phi+1)/2) + 0.5*log((1-phi)/2);
    CurrentLJL = CurrentLJL + Prior;
    
    
    HPNew      = [beta sigma phi];
    
    OriginalHP    = HPNew;
    OriginalHP(2) = log(OriginalHP(2));
    OriginalHP(3) = atanh(OriginalHP(3));
    
    HPProposed = HPProposed + 1;
    
    
    % Create Gradient
    HPGradL(1) = -NumOfObs/beta + sum( (ySquared)./(beta^3*Expx) );
    HPGradL(2) = -NumOfObs + x(1)^2*(1-phi^2)/(sigma^2) + sum( ((x(2:end)-phi*x(1:end-1)).^2)/(sigma^2) );
    HPGradL(3) = -phi + (phi*x(1)^2)*(1-phi^2)/sigma^2 + sum( x(1:end-1).*(x(2:end)-phi*x(1:end-1)).*(1-phi^2)./sigma^2 );
    % Add Prior
    HPGradL(1) = HPGradL(1) - 1;
    HPGradL(2) = HPGradL(2) + 0.5/sigma^2 - 11;
    HPGradL(3) = HPGradL(3) + 38*(1-phi) - (1+phi);
    
    % Fisher Information & Metric Tensor
    G = zeros(3);
    G(1,1) = (2*NumOfObs)/beta^2;
    G(2,2) = (2*NumOfObs);
    G(2,3) = 2*phi;
    G(3,2) = G(2,3);
    G(3,3) = 2*phi^2 - (NumOfObs-1)*(phi^2 - 1);
    % Add Prior
    PriorG = zeros(3);
    PriorG(2,2) = -1/sigma^2;
    PriorG(3,3) = (-38-1)*(1-phi^2);
    
    G = G - PriorG;
    
    InvG = inv(G);
    
    % w.r.t. beta
    dGdParas{1} = zeros(3);
    dGdParas{1}(1,1) = (-4*NumOfObs)/beta^3;
    % w.r.t. sigma
    dGdParas{2} = zeros(3);
    % Add Prior w.r.t. sigma
    dGdParas{2}(2,2) = dGdParas{2}(2,2) - (2/sigma^2);
    % w.r.t. phi
    dGdParas{3} = zeros(3);
    dGdParas{3}(2,3) = 2*(1-phi^2);
    dGdParas{3}(3,2) = dGdParas{3}(2,3);
    dGdParas{3}(3,3) = ((4*phi) - (NumOfObs-1)*(2*phi))*(1-phi^2);
    % Add Prior w.r.t. phi
    dGdParas{3}(3,3) = dGdParas{3}(3,3) - (4*phi)*(20+1.5-2)*(1-phi^2);
    
    for i = 1:3
        InvGdG{i}         = G\dGdParas{i};
        TraceInvGdG(i)    = trace(InvGdG{i});
        HPSecondTerm(:,i) = InvGdG{i}*InvG(:,i);
    end
    
    
    % Update parameters
    HPNew(2) = log(HPNew(2));
    HPNew(3) = atanh(HPNew(3));
    
    Mean  = HPNew + (HPStepSize/(2))*(G\HPGradL')'...
                  - HPStepSize*sum(HPSecondTerm,2)'... % Sum across columns
                  + (HPStepSize/(2))*(G\TraceInvGdG')';
    
    try
        HPNew = Mean + ( randn(1,3)*chol(HPStepSize*InvG + eye(3)*1e-8) );
    catch
        disp('error')
    end
    
    % Calculate prob of New given Old
    ProbNewGivenOld = -sum(log(diag(HPStepSize*InvG))) - 0.5*(Mean-HPNew)*(G/HPStepSize)*(Mean-HPNew)';
    
    HPNew(2) = exp(HPNew(2));
    HPNew(3) = tanh(HPNew(3));
    
    % Calculate proposed likelihood
    ProposedLJL = -sum(x/2) - NumOfObs*log(HPNew(1)) - sum( (ySquared)./(2*HPNew(1)^2*exp(x)) ) + 0.5*log(1-HPNew(3)^2) - log(HPNew(2)) - x(1)^2*(1-HPNew(3)^2)/(2*HPNew(2)^2) - (NumOfObs-1)*log(HPNew(2)) - sum( (x(2:end)-HPNew(3)*x(1:end-1)).^2./(2*HPNew(2)^2) );
    ProposedJ   = log(HPNew(2)*(1-HPNew(3)^2)); % Jacobian
    % Add Prior
    Prior       = -HPNew(1) - 0.5/(2*HPNew(2)^2) - 6*log(HPNew(2)^2) + log(HPNew(2)) + 19*log((HPNew(3)+1)/2) + 0.5*log((1-HPNew(3))/2);
    ProposedLJL = ProposedLJL + Prior;
    
    
    % Calculate new gradient and metric tensor
    % Precalculate exp(x) for speed
    Expx = exp(x);
    
    % Create Gradient
    HPGradL(1) = -NumOfObs/HPNew(1) + sum( (ySquared)./(HPNew(1)^3*Expx) );
    HPGradL(2) = -NumOfObs + x(1)^2*(1-HPNew(3)^2)/(HPNew(2)^2) + sum( ((x(2:end)-HPNew(3)*x(1:end-1)).^2)/(HPNew(2)^2) );
    HPGradL(3) = -HPNew(3) + (HPNew(3)*x(1)^2)*(1-HPNew(3)^2)/HPNew(2)^2 + sum( x(1:end-1).*(x(2:end)-HPNew(3)*x(1:end-1)).*(1-HPNew(3)^2)./HPNew(2)^2 );
    % Add Prior
    HPGradL(1) = HPGradL(1) - 1;
    HPGradL(2) = HPGradL(2) + 0.5/HPNew(2)^2 - 11;
    HPGradL(3) = HPGradL(3) + 38*(1-HPNew(3)) - (1+HPNew(3));
    
    % Calculate G
    G = zeros(3);
    G(1,1) = (2*NumOfObs)/HPNew(1)^2;
    G(2,2) = (2*NumOfObs);
    G(2,3) = 2*HPNew(3);
    G(3,2) = G(2,3);
    G(3,3) = 2*HPNew(3)^2 - (NumOfObs-1)*(HPNew(3)^2 - 1);
    % Add Prior
    PriorG = zeros(3);
    PriorG(2,2) = -1/HPNew(2)^2;
    PriorG(3,3) = (-38-1)*(1-HPNew(3)^2);
    
    G = G - PriorG;
    
    InvG = inv(G);
    
    % w.r.t. beta
    dGdParas{1} = zeros(3);
    dGdParas{1}(1,1) = (-4*NumOfObs)/HPNew(1)^3;
    % w.r.t. sigma
    dGdParas{2} = zeros(3);
    % Add Prior w.r.t. sigma
    dGdParas{2}(2,2) = dGdParas{2}(2,2) - (2/HPNew(2)^2);
    % w.r.t. phi
    dGdParas{3} = zeros(3);
    dGdParas{3}(2,3) = 2*(1-HPNew(3)^2);
    dGdParas{3}(3,2) = dGdParas{3}(2,3);
    dGdParas{3}(3,3) = ((4*HPNew(3)) - (NumOfObs-1)*(2*HPNew(3)))*(1-HPNew(3)^2);
    % Add Prior w.r.t. phi
    dGdParas{3}(3,3) = dGdParas{3}(3,3) - (4*HPNew(3))*(20+1.5-2)*(1-HPNew(3)^2);
    
    
    for i = 1:3
        InvGdG{i}         = G\dGdParas{i};
        TraceInvGdG(i)    = trace(InvGdG{i});
        HPSecondTerm(:,i) = InvGdG{i}*InvG(:,i);
    end
    
    HPNew(2) = log(HPNew(2));
    HPNew(3) = atanh(HPNew(3));
    
    % Calculate prob of Old given New
    Mean  = HPNew + (HPStepSize/(2))*(G\HPGradL')'...
                  - HPStepSize*sum(HPSecondTerm,2)'... % Sum across columns
                  + (HPStepSize/(2))*(InvG*TraceInvGdG')';
              
    ProbOldGivenNew = -sum(log(diag(HPStepSize*InvG))) - 0.5*(Mean-OriginalHP)*(G/HPStepSize)*(Mean-OriginalHP)';
    
    HPNew(2) = exp(HPNew(2));
    HPNew(3) = tanh(HPNew(3));
    
    % Accept according to ratio
    Ratio = ProposedLJL + ProposedJ + ProbOldGivenNew - CurrentLJL - CurrentJ - ProbNewGivenOld;

    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        beta  = HPNew(1);
        sigma = HPNew(2);
        phi   = HPNew(3);
        HPAccepted = HPAccepted + 1;
        %disp('Accepted')
        drawnow
    else
        %disp('Rejected')
        drawnow
    end
        
    
    
    
    if mod(IterationNum, 100) == 0
        Acceptance   = Accepted/Proposed;
        HPAcceptance = HPAccepted/HPProposed;
        
        disp(['Hyperparameter Acceptance: ' num2str(HPAcceptance)])
        disp(['X Acceptance: ' num2str(Acceptance)])
        
        
        Proposed = 0;
        Accepted = 0;
        
        HPProposed = 0;
        HPAccepted = 0;
    end
        

    % Save samples if required
    if IterationNum > BurnIn
        xSaved(IterationNum-BurnIn,:)     = x;
        betaSaved(IterationNum-BurnIn,1)  = beta;
        phiSaved(IterationNum-BurnIn,1)   = phi;
        sigmaSaved(IterationNum-BurnIn,1) = sigma;
        LJLSaved(IterationNum-BurnIn)     = CurrentLJL;
    end
    
    % Start timer after burn-in
    if IterationNum == BurnIn
        disp('Burn-in complete, now drawing posterior samples.')
        tic;
    end
    
end

% Stop timer
TimeTaken = toc;


CurTime = fix(clock);
save(['Results/mMALA_Trans_' num2str(StepSize)  '_StochVol_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'xSaved', 'Truex', 'y', 'betaSaved', 'phiSaved', 'sigmaSaved', 'LJLSaved', 'TimeTaken', 'StepSize', 'HPStepSize');

figure(100)
subplot(231)
plot(betaSaved)
subplot(232)
plot(sigmaSaved)
subplot(233)
plot(phiSaved)
% Plot histograms
%
subplot(234)
hist(betaSaved,1000)
subplot(235)
hist(sigmaSaved,1000)
subplot(236)
hist(phiSaved,1000)
%}

end


