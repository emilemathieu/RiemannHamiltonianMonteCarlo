function [] = StochVol_MALA()

% Random Numbers...
randn('state', sum(100*clock));
rand('twister', sum(100*clock));

% True Parameters
beta  = 0.65;
sigma = 0.15;
phi   = 0.98;


%%%%% Load Saved Data %%%%%
%{
NumOfObs = 2000;
x = zeros(2000,1);

% Generate latent volatility at t = 1
x(1) = normrnd(0, sigma/sqrt(1 - phi^2));

% Generate latent volatilities
for n = 1:(NumOfObs-1)
    x(n + 1) = phi*x(n) + normrnd(0, sigma);
end

% Generate observed data
y = beta*randn(NumOfObs,1).*exp(x/2);
I           = eye(NumOfObs);

Truex = x; % Save true x values
%}


%%%%% Load Saved Data %%%%%
%
Data = load('StochVolData1.mat');

y        = Data.y;
NumOfObs = length(y);
x        = Data.Truex;
Truex    = Data.Truex;
%}



% Sparse diagonal matrix
I = sparse(eye(NumOfObs));

subplot(121);
plot(y)
subplot(122)
plot(x)



% RM-HMC Setup
NumOfIterations    = 100000;
BurnIn             = 80000;
Thinning           = 1;

% Values for StochVolData1
HPStepSize         = 0.01;
StepSize           = 0.05;


Scaling   = NumOfObs^(1/2);
HPScaling = NumOfObs^(1/2);

Proposed = 0;
Accepted = 0;

HPProposed = 0;
HPAccepted = 0;


% Set initial values of x, beta, sigma, phi
x     = y;
beta  = 0.5;
sigma = 0.5;
phi   = 0.5;

ySquared = y.^2;

xSaved     = zeros((NumOfIterations-BurnIn)/Thinning,NumOfObs);
betaSaved  = zeros((NumOfIterations-BurnIn)/Thinning,1);
phiSaved   = zeros((NumOfIterations-BurnIn)/Thinning,1);
sigmaSaved = zeros((NumOfIterations-BurnIn)/Thinning,1);
LJLSaved   = zeros((NumOfIterations-BurnIn)/Thinning,1);



for IterationNum = 1:NumOfIterations
       
    if mod(IterationNum, 50) == 0
        disp(IterationNum)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample the latent variables %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Update the log-jointlikelihood!
    CurrentLJL =  -(1/(2*sigma^2))*x(1)^2*(1-phi^2) - sum( x./2 + (y.^2)./(2*beta^2*exp(x)) ) - (1/(2*sigma^2))*sum( (x(2:end)-phi*x(1:end-1)).^2 );
    
    xNew = x;
    Proposed = Proposed + 1;
    
    % Create Gradient
    s   = -0.5 + (y.^2)./(2*beta^2*exp(xNew));
    d_1 = (1/sigma^2)*(xNew(1) - phi*xNew(2));
    d_2 = (1/sigma^2)*(xNew(end) - phi*xNew(end-1));
    r   = [d_1; -(phi/sigma^2)*(xNew(3:end) - phi*xNew(2:end-1)) + (1/sigma^2)*(xNew(2:end-1) - phi*xNew(1:end-2)); d_2];
    
    % Gradient
    GradL = s - r;
    
    Mean = xNew + (StepSize/(2*Scaling))*GradL;
    xNew = Mean + randn(NumOfObs,1)*sqrt((StepSize/Scaling));
    
    ProposedLJL   = -(1/(2*sigma^2))*xNew(1)^2*(1-phi^2) - sum( xNew./2 + (y.^2)./(2*beta^2*exp(xNew)) ) - (1/(2*sigma^2))*sum( (xNew(2:end)-phi*xNew(1:end-1)).^2 );
    
    
    ProbNewGivenOld = LogNormPDF(Mean, xNew,(StepSize/Scaling));
    
    
    % Calculate probability of Old given New
    % Create Gradient
    s   = -0.5 + (y.^2)./(2*beta^2*exp(xNew));
    d_1 = (1/sigma^2)*(xNew(1) - phi*xNew(2));
    d_2 = (1/sigma^2)*(xNew(end) - phi*xNew(end-1));
    r   = [d_1; -(phi/sigma^2)*(xNew(3:end) - phi*xNew(2:end-1)) + (1/sigma^2)*(xNew(2:end-1) - phi*xNew(1:end-2)); d_2];
    % Gradient
    GradL = s - r;
    
    
    Mean = xNew + (StepSize/(2*Scaling))*GradL;
    ProbOldGivenNew = LogNormPDF(x, Mean, (StepSize/Scaling));
    
    
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
    % Sample Parameters %
    %%%%%%%%%%%%%%%%%%%%%
    
    Expx = exp(x);
    
    
    HPNew      = [beta sigma phi];
    
    OriginalHP    = HPNew;
    OriginalHP(2) = log(OriginalHP(2));
    OriginalHP(3) = atanh(OriginalHP(3));
    
    HPProposed = HPProposed + 1;
    
    CurrentLJL = -sum(x/2) - NumOfObs*log(beta) - sum( (ySquared)./(2*beta^2*Expx) ) + 0.5*log(1-phi^2) - log(sigma) - x(1)^2*(1-phi^2)/(2*sigma^2) - (NumOfObs-1)*log(sigma) - sum( (x(2:end)-phi*x(1:end-1)).^2./(2*sigma^2) );
    CurrentJ   = log(sigma*(1-phi^2)); % Jacobian
    % Add Prior
    Prior      = -beta - 0.5/(2*sigma^2) - 6*log(sigma^2) + log(sigma) + 19*log((phi+1)/2) + 0.5*log((1-phi)/2);
    CurrentLJL = CurrentLJL + Prior;
    
    
    % Create Gradient
    HPGradL(1) = -NumOfObs/beta + sum( (ySquared)./(beta^3*Expx) );
    HPGradL(2) = -NumOfObs + x(1)^2*(1-phi^2)/(sigma^2) + sum( ((x(2:end)-phi*x(1:end-1)).^2)/(sigma^2) );
    HPGradL(3) = -phi + (phi*x(1)^2)*(1-phi^2)/sigma^2 + sum( x(1:end-1).*(x(2:end)-phi*x(1:end-1)).*(1-phi^2)./sigma^2 );
    % Add Prior
    HPGradL(1) = HPGradL(1) - 1;
    HPGradL(2) = HPGradL(2) + 0.5/sigma^2 - 11;
    HPGradL(3) = HPGradL(3) + 38*(1-phi) - (1+phi);
    
    % Update parameters
    HPNew(2) = log(HPNew(2));
    HPNew(3) = atanh(HPNew(3));
    
    HPMean = HPNew + (HPStepSize/(2*HPScaling))*HPGradL;
    HPNew  = HPMean + randn(1,3)*sqrt((HPStepSize/HPScaling));
    
    ProbNewGivenOld = LogNormPDF(HPMean, HPNew,(HPStepSize/HPScaling));
    
    HPNew(2) = exp(HPNew(2));
    HPNew(3) = tanh(HPNew(3));
    
    
    ProposedLJL = -sum(x/2) - NumOfObs*log(HPNew(1)) - sum( (ySquared)./(2*HPNew(1)^2*exp(x)) ) + 0.5*log(1-HPNew(3)^2) - log(HPNew(2)) - x(1)^2*(1-HPNew(3)^2)/(2*HPNew(2)^2) - (NumOfObs-1)*log(HPNew(2)) - sum( (x(2:end)-HPNew(3)*x(1:end-1)).^2./(2*HPNew(2)^2) );
    ProposedJ   = log(HPNew(2)*(1-HPNew(3)^2)); % Jacobian
    % Add Prior
    Prior       = -HPNew(1) - 0.5/(2*HPNew(2)^2) - 6*log(HPNew(2)^2) + log(HPNew(2)) + 19*log((HPNew(3)+1)/2) + 0.5*log((1-HPNew(3))/2);
    ProposedLJL = ProposedLJL + Prior;
    
    
    % Create Gradient
    HPGradL(1) = -NumOfObs/HPNew(1) + sum( (ySquared)./(HPNew(1)^3*Expx) );
    HPGradL(2) = -NumOfObs + x(1)^2*(1-HPNew(3)^2)/(HPNew(2)^2) + sum( ((x(2:end)-HPNew(3)*x(1:end-1)).^2)/(HPNew(2)^2) );
    HPGradL(3) = -HPNew(3) + (HPNew(3)*x(1)^2)*(1-HPNew(3)^2)/HPNew(2)^2 + sum( x(1:end-1).*(x(2:end)-HPNew(3)*x(1:end-1)).*(1-HPNew(3)^2)./HPNew(2)^2 );
    % Add Prior
    HPGradL(1) = HPGradL(1) - 1;
    HPGradL(2) = HPGradL(2) + 0.5/HPNew(2)^2 - 11;
    HPGradL(3) = HPGradL(3) + 38*(1-HPNew(3)) - (1+HPNew(3));
    
    % Calculate probability of Old given New
    HPNew(2) = log(HPNew(2));
    HPNew(3) = atanh(HPNew(3));
    
    HPMean          = HPNew + (HPStepSize/(2*HPScaling))*HPGradL;
    ProbOldGivenNew = LogNormPDF(OriginalHP, HPMean, (HPStepSize/HPScaling));
    
    HPNew(2) = exp(HPNew(2));
    HPNew(3) = tanh(HPNew(3));
    
    % Accept according to ratio
    Ratio = ProposedLJL + ProposedJ + ProbOldGivenNew - CurrentLJL - CurrentJ - ProbNewGivenOld;
    
    
    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        beta       = HPNew(1);
        sigma      = HPNew(2);
        phi        = HPNew(3);
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
        
        
        disp(['HP: ' num2str(HPAcceptance)])
        disp(['LV: ' num2str(Acceptance)])
        
        
        Proposed = 0;
        Accepted = 0;
        
        HPProposed = 0;
        HPAccepted = 0;
    end
    
    
    % Save samples if required
    if IterationNum > BurnIn && mod(IterationNum, Thinning) == 0
        xSaved((IterationNum-BurnIn)/Thinning,:)     = x;
        betaSaved((IterationNum-BurnIn)/Thinning,1)  = beta;
        phiSaved((IterationNum-BurnIn)/Thinning,1)   = phi;
        sigmaSaved((IterationNum-BurnIn)/Thinning,1) = sigma;
        LJLSaved((IterationNum-BurnIn)/Thinning)     = CurrentLJL;
    end
    
    % Start timer after burn-in
    if IterationNum == BurnIn
        disp('Burn-in complete, now drawing posterior samples.')
        
        HPStepSize         = 0.005;
        StepSize           = 0.03;

        Scaling   = NumOfObs^(1/3);
        HPScaling = NumOfObs^(1/3);
        
        tic;
    end
    
end

% Stop timer
TimeTaken = toc;


CurTime = fix(clock);
save(['Results/MALA_StochVol_Trans_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'xSaved', 'Truex', 'y', 'betaSaved', 'phiSaved', 'sigmaSaved', 'LJLSaved', 'TimeTaken')

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


