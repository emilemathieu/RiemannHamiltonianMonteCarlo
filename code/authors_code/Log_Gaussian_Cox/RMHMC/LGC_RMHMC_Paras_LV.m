function [] = LGC_RMHMC_Paras_LV()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up path for lightspeed toolbox %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('/scratch/bc/Software/lightspeed'))
%addpath(genpath('/Applications/Matlab_Addons/lightspeed'))

% Randomize
randn('state', sum(clock) );
rand('twister', sum(clock) );

% Grid Size
N     = 64;

% Load data
% Y is the data
% X are the latent variables
Data=load('TestData64.mat');
y = Data.Y;


% Hyperparameters of model - To be inferred
NumOfHyperparameters = 2;
sigmaSq = 1.91;
Beta    = 1/33;

Mu    = log(126) - sigmaSq/2; % Fixed
m     = 1/(N^2);              % Fixed

% Set up Gamma Prior Parameters
Gammak     = 2;
GammaTheta = 0.5;

[D] = length(y);

% RMHMC Setup
NumOfIterations    = 6000;
BurnIn             = 1000;

HPNumOfLeapFrogSteps           = 1;
HPStepSize                     = 0.2;
NumOfHPFixedPointSteps         = 3;
NumOfHPFixedPointStepsMomentum = 10;


LVNumOfLeapFrogSteps = 20;
LVStepSize           = 0.1;

% Counters
HPProposed = 0;
HPAccepted = 0;
LVProposed = 0;
LVAccepted = 0;

% Set initial values of w
muOnes   = ones(D,1)*Mu;

% Start from mu
x        = muOnes;

LVSaved    = zeros(NumOfIterations-BurnIn, D);
HPSaved    = zeros(NumOfIterations-BurnIn, NumOfHyperparameters);
LVLJLSaved = zeros(NumOfIterations-BurnIn, 1);
HPLJLSaved = zeros(NumOfIterations-BurnIn, 1);

% Calculate covariance matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SigmaInv       = zeros(N^2);
Sigma          = zeros(N^2);
Dist           = zeros(N^2);
x_r            = 0:1/(N - 1):1;
y_r            = 0:1/(N - 1):1;
[xs, ys]       = meshgrid(x_r, y_r);
coord_xy(:, 1) = xs(:);
coord_xy(:, 2) = ys(:);

%%% FASTER %%%
for i=1:N^2,
    coords1   = repmat(coord_xy(i,:), N^2, 1) - coord_xy; 
    Dist(i,:) = sum(coords1.^2, 2).^0.5;
end
Sigma = sigmaSq.*exp( -Dist./(Beta*N) );

SigmaInv = chol2inv(chol(Sigma));

% Free up memory
coords1  = [];
coord_xy = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LVCurrentLJL =  y'*x - sum(m*exp(x)) - 0.5*(x - muOnes)'*SigmaInv*(x - muOnes);

HPCurrentLJL = -0.5*( log(2) + NumOfHyperparameters*log(pi) + 2*sum(log(diag(chol(Sigma)))) ) - 0.5*(x - muOnes)'*SigmaInv*(x - muOnes);
HPCurrentLJL = HPCurrentLJL + (Gammak-1)*log(sigmaSq) - sigmaSq/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for sigmaSq
HPCurrentLJL = HPCurrentLJL + (Gammak-1)*log(Beta) - Beta/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for Beta




% Caluclate first and second partial derivatives of covariance function
dSdsigmaSqT        = sigmaSq*exp(-Dist./(N*Beta));
dSdBetaT           = Dist./(N*Beta).*dSdsigmaSqT;
d2SdBetaT2         = -dSdBetaT + Dist./(N*Beta).*dSdBetaT;

% Pre-calculate useful expressions
SInvdSdsigmaSqT        = SigmaInv*dSdsigmaSqT;
SInvdSdBetaT           = SigmaInv*dSdBetaT;
SInvdSdsigmaSqTAllSq   = SInvdSdsigmaSqT*SInvdSdsigmaSqT;
SInvdSdBetaTAllSq      = SInvdSdBetaT*SInvdSdBetaT;
SInvd2SdBetaT2         = SigmaInv*d2SdBetaT2;
    
% Calculate G
CurrentG = [];
CurrentG(1,1) = 0.5*sum(sum(SInvdSdsigmaSqT.*SInvdSdsigmaSqT'));
CurrentG(1,2) = 0.5*sum(sum(SInvdSdsigmaSqT.*SInvdSdBetaT'));
CurrentG(2,1) = CurrentG(1,2);
CurrentG(2,2) = 0.5*sum(sum(SInvdSdBetaT.*SInvdSdBetaT'));
% Subtract the second partial derivatives of prior
CurrentG(1,1) = CurrentG(1,1) - (-sigmaSq/GammaTheta);
CurrentG(2,2) = CurrentG(2,2) - (-Beta/GammaTheta);

CurrentCholG = chol(CurrentG);
CurrentInvG  = chol2inv(CurrentCholG);
    
%%% Calculate partial derivatives of metric tensor %%%
CurrentdGdsigmaSq(1,1) = -sum(sum( SInvdSdsigmaSqTAllSq.*SInvdSdsigmaSqT' )) + trace( SInvdSdsigmaSqTAllSq );
CurrentdGdsigmaSq(1,2) = -sum(sum( SInvdSdsigmaSqTAllSq.*SInvdSdBetaT' )) + sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaT' ));
CurrentdGdsigmaSq(2,1) = CurrentdGdsigmaSq(1,2);
CurrentdGdsigmaSq(2,2) = -sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaTAllSq' )) + trace( SInvdSdBetaTAllSq );
CurrentdGdsigmaSq(1,1) = CurrentdGdsigmaSq(1,1) + sigmaSq/GammaTheta;


CurrentdGdBeta(1,1) = -sum(sum( SInvdSdBetaT.*SInvdSdsigmaSqTAllSq' )) + sum(sum( SInvdSdBetaT.*SInvdSdsigmaSqT' ));
CurrentdGdBeta(1,2) = -sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaTAllSq' )) + 0.5*trace( SInvdSdBetaTAllSq ) + 0.5*sum(sum( SInvdSdsigmaSqT.*SInvd2SdBetaT2' ));
CurrentdGdBeta(2,1) = CurrentdGdBeta(1,2);
CurrentdGdBeta(2,2) = -sum(sum( SInvdSdBetaTAllSq.*SInvdSdBetaT' )) + sum(sum( SInvd2SdBetaT2.*SInvdSdBetaT' ));
CurrentdGdBeta(2,2) = CurrentdGdBeta(2,2) + Beta/GammaTheta;



for IterationNum = 1:NumOfIterations
       
    disp(IterationNum)
    
    
    
    if mod(IterationNum,50) == 0
        disp([num2str(IterationNum) ' iterations completed.'])
        disp('HP Acceptance Rate:')
        disp(HPAccepted/HPProposed)
        HPProposed = 0;
        HPAccepted = 0;
        
        disp('LV Acceptance Rate:')
        disp(LVAccepted/LVProposed)
        LVProposed = 0;
        LVAccepted = 0;
        
        drawnow
    end
    
    %disp(['Iteration ' num2str(IterationNum)])
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample the Hyperparameters %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    HPCurrentLJL = -0.5*( log(2) + NumOfHyperparameters*log(pi) + 2*sum(log(diag(chol(Sigma)))) ) - 0.5*(x - muOnes)'*SigmaInv*(x - muOnes);
    HPCurrentLJL = HPCurrentLJL + (Gammak-1)*log(sigmaSq) - sigmaSq/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for sigmaSq
    HPCurrentLJL = HPCurrentLJL + (Gammak-1)*log(Beta) - Beta/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for Beta
    HPCurrentJ   = log(sigmaSq*Beta); % Jacobian
    
    HPNew      = [log(sigmaSq) log(Beta)];
    HPProposed = HPProposed + 1;
    
    
    % Caluclate first and second partial derivatives of covariance function
    dSdsigmaSqT        = sigmaSq*exp(-Dist./(N*Beta));
    dSdBetaT           = Dist./(N*Beta).*dSdsigmaSqT;
    
    % Calculate Gradient
    TempVec = (x - muOnes)'*SigmaInv;
    
    %GradL(1) = -0.5*trace( SigmaInv*dSdsigmaSq ) + 0.5*(x - muOnes)'*SigmaInv*dSdsigmaSq*SigmaInv*(x - muOnes);
    %GradL(2) = -0.5*trace( SigmaInv*dSdBeta ) + 0.5*(x - muOnes)'*SigmaInv*dSdBeta*SigmaInv*(x - muOnes);
    GradL = [];
    GradL(1) = -0.5*sum(sum( SigmaInv.*dSdsigmaSqT' )) + 0.5*TempVec*dSdsigmaSqT*TempVec';
    GradL(1) = GradL(1) + (Gammak-1) - sigmaSq/GammaTheta; % Add derivative of prior wrt parameter sigmaSq
    GradL(2) = -0.5*sum(sum( SigmaInv.*dSdBetaT' )) + 0.5*(x - muOnes)'*SigmaInv*dSdBetaT*SigmaInv*(x - muOnes);
    GradL(2) = GradL(2) + (Gammak-1) - Beta/GammaTheta; % Add derivative of prior wrt parameter Beta
    
    
    G          = CurrentG;
    CholG      = CurrentCholG;
    InvG       = CurrentInvG;
    dGdsigmaSq = CurrentdGdsigmaSq;
    dGdBeta    = CurrentdGdBeta;
    
    InvGdG{1}      = InvG*dGdsigmaSq;
    InvGdG{2}      = InvG*dGdBeta;
    
    % Calculate trace of invG * dGdParas
    TraceInvGdGdParas(1) = trace(InvGdG{1});
    TraceInvGdGdParas(2) = trace(InvGdG{2});
    
    TraceTerm = 0.5*TraceInvGdGdParas';
    
    ProposedMomentum = (randn(1,NumOfHyperparameters)*CurrentCholG)';
    OriginalMomentum = ProposedMomentum;
        
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
        
    RandomSteps = ceil(rand*HPNumOfLeapFrogSteps);
        
    % Perform leapfrog steps
    for StepNum = 1:RandomSteps
        
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%        
        % Multiple fixed point iteration
        PM = ProposedMomentum;
        for FixedIter = 1:NumOfHPFixedPointStepsMomentum
            MomentumHist(FixedIter,:) = PM;

            InvGMomentum = InvG*PM;
            for d = 1:NumOfHyperparameters
                LastTerm(d)  = 0.5*(PM'*InvGdG{d}*InvGMomentum);
            end

            PM = ProposedMomentum + TimeStep*(HPStepSize/2)*(GradL' - TraceTerm + LastTerm');
        end
        ProposedMomentum = PM;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%
        % Update w parameters %
        %%%%%%%%%%%%%%%%%%%%%%%
        PHP = HPNew;
        SigmaInvBreak = false;
        
        for FixedIter = 1:NumOfHPFixedPointSteps
            wHist(FixedIter,:) = PHP;
        
            InvGMomentum         = G\ProposedMomentum;
            
            % Save if it is 1st iteration
            if FixedIter == 1
                OriginalInvGMomentum = InvGMomentum;
            end
            
            PHP = HPNew + (TimeStep*(HPStepSize/2))*OriginalInvGMomentum' + (TimeStep*(HPStepSize/2))*InvGMomentum';
            
            sigmaSqNew = exp(PHP(1));
            BetaNew    = exp(PHP(2));

            SigmaNew    = sigmaSqNew.*exp( -Dist./(BetaNew*N) );
            try
                SigmaInvNew = chol2inv(chol(SigmaNew));
            catch
                SigmaInvBreak = true;
                break
            end
            
            % Caluclate first and second partial derivatives of covariance function
            dSdsigmaSqT        = sigmaSqNew*exp(-Dist./(N*BetaNew));
            dSdBetaT           = Dist./(N*BetaNew).*dSdsigmaSqT;
            d2SdBetaT2         = -dSdBetaT + Dist./(N*BetaNew).*dSdBetaT;
        
            % Pre-calculate useful expressions
            SInvdSdsigmaSqT        = SigmaInv*dSdsigmaSqT;
            SInvdSdBetaT           = SigmaInv*dSdBetaT;
            
            % Calculate G
            G(1,1) = 0.5*sum(sum(SInvdSdsigmaSqT.*SInvdSdsigmaSqT')); %0.5*trace(SInvdSdsigmaSqAllSq);
            G(1,2) = 0.5*sum(sum(SInvdSdsigmaSqT.*SInvdSdBetaT')); %0.5*sum(sum( SInvdSdsigmaSq.*SInvdSdBeta' ));
            G(2,1) = G(1,2);
            G(2,2) = 0.5*sum(sum(SInvdSdBetaT.*SInvdSdBetaT')); %0.5*trace(SInvdSdBetaAllSq);
            % Subtract negative second derivative
            G(1,1) = G(1,1) - (-sigmaSqNew/GammaTheta);
            G(2,2) = G(2,2) - (-BetaNew/GammaTheta);

        end
        HPNew = PHP;
        
        if SigmaInvBreak
            break
        end
        
        
        sigmaSqNew = exp(HPNew(1));
        BetaNew    = exp(HPNew(2));
        
        
        % Pre-calculate useful expressions
        SInvdSdsigmaSqTAllSq   = SInvdSdsigmaSqT*SInvdSdsigmaSqT;
        SInvdSdBetaTAllSq      = SInvdSdBetaT*SInvdSdBetaT;
        SInvd2SdBetaT2         = SigmaInv*d2SdBetaT2;

        % Calculate Gradient
        TempVec = (x - muOnes)'*SigmaInvNew;
        
        GradL = [];
        GradL(1) = -0.5*sum(sum( SigmaInvNew.*dSdsigmaSqT' )) + 0.5*TempVec*dSdsigmaSqT*TempVec';
        GradL(1) = GradL(1) + (Gammak-1) - sigmaSqNew/GammaTheta; % Add derivative of prior wrt parameter sigmaSq
        GradL(2) = -0.5*sum(sum( SigmaInvNew.*dSdBetaT' )) + 0.5*TempVec*dSdBetaT*TempVec';
        GradL(2) = GradL(2) + (Gammak-1) - BetaNew/GammaTheta; % Add derivative of prior wrt parameter Beta
        
        CholG  = chol(G);
        InvG   = chol2inv(CholG);

        %%% Calculate partial derivatives of metric tensor %%%
        dGdsigmaSq(1,1) = -sum(sum( SInvdSdsigmaSqTAllSq.*SInvdSdsigmaSqT' )) + trace( SInvdSdsigmaSqTAllSq );
        dGdsigmaSq(1,2) = -sum(sum( SInvdSdsigmaSqTAllSq.*SInvdSdBetaT' )) + sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaT' ));
        dGdsigmaSq(2,1) = dGdsigmaSq(1,2);
        dGdsigmaSq(2,2) = -sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaTAllSq' )) + trace( SInvdSdBetaTAllSq );
        dGdsigmaSq(1,1) = dGdsigmaSq(1,1) + sigmaSqNew/GammaTheta;
        
        dGdBeta(1,1) = -sum(sum( SInvdSdBetaT.*SInvdSdsigmaSqTAllSq' )) + sum(sum( SInvdSdBetaT.*SInvdSdsigmaSqT' ));
        dGdBeta(1,2) = -sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaTAllSq' )) + 0.5*trace( SInvdSdBetaTAllSq ) + 0.5*sum(sum( SInvdSdsigmaSqT.*SInvd2SdBetaT2' ));
        dGdBeta(2,1) = dGdBeta(1,2);
        dGdBeta(2,2) = -sum(sum( SInvdSdBetaTAllSq.*SInvdSdBetaT' )) + sum(sum( SInvd2SdBetaT2.*SInvdSdBetaT' ));
        dGdBeta(2,2) = dGdBeta(2,2) + BetaNew/GammaTheta;

        InvGdG{1}    = InvG*dGdsigmaSq;
        InvGdG{2}    = InvG*dGdBeta;

        % Calculate trace of invG * dGdParas
        TraceInvGdGdParas(1) = trace(InvGdG{1});
        TraceInvGdGdParas(2) = trace(InvGdG{2});

        TraceTerm = 0.5*TraceInvGdGdParas';
        
        
            
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        InvGMomentum = InvG*ProposedMomentum;
        for d = 1:NumOfHyperparameters
            LastTerm(d)  = 0.5*(ProposedMomentum'*InvGdG{d}*InvGMomentum);
        end
        
        ProposedMomentum = ProposedMomentum + (TimeStep*HPStepSize/2)*(GradL' - TraceTerm + LastTerm');
        
    end

    
    % Calculate proposed H value
    try
        if SigmaInvBreak == false

            ProposedLJL    = - 0.5*( log(2) + NumOfHyperparameters*log(pi) + 2*sum(log(diag(chol(SigmaNew)))) ) - 0.5*(x - muOnes)'*SigmaInvNew*(x - muOnes);
            ProposedLJL    = ProposedLJL + (Gammak-1)*log(sigmaSqNew) - sigmaSqNew/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for sigmaSq
            ProposedLJL    = ProposedLJL + (Gammak-1)*log(BetaNew) - BetaNew/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for Beta
            HPProposedJ    = log(sigmaSqNew*BetaNew); % Jacobian
            
            ProposedLogDet = 0.5*( log(2) + NumOfHyperparameters*log(pi) + 2*sum(log(diag(CholG))) );
            ProposedH      = -ProposedLJL - HPProposedJ + ProposedLogDet + (ProposedMomentum'*InvG*ProposedMomentum)/2;

            % Calculate current H value
            CurrentLogDet  = 0.5*( log(2) + NumOfHyperparameters*log(pi) + 2*sum(log(diag(CurrentCholG))) );
            CurrentH       = -HPCurrentLJL - HPCurrentJ + CurrentLogDet + (OriginalMomentum'*CurrentInvG*OriginalMomentum)/2;

            % Accept according to ratio
            Ratio = -ProposedH + CurrentH;

            if Ratio > 0 || (Ratio > log(rand))
                HPCurrentLJL = ProposedLJL;
                sigmaSq      = exp(HPNew(1));
                Beta         = exp(HPNew(2));

                CurrentG          = G;
                CurrentCholG      = CholG;
                CurrentInvG       = InvG;
                CurrentdGdsigmaSq = dGdsigmaSq;
                CurrentdGdBeta    = dGdBeta;

                Sigma    = SigmaNew;
                SigmaInv = SigmaInvNew;

                HPAccepted = HPAccepted + 1;
                disp('Hyperparameter Step Accepted')
                drawnow
            else
                disp('Hyperparameter Step Rejected')
                disp(Ratio)
                disp([num2str(sigmaSq) ' ' num2str(Beta)])
                drawnow
            end
        else
            SigmaInvBreak = false;
        end
    catch
        disp('Rejected')
        drawnow
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample the latent variables %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    xNew           = x;
    LVProposed     = LVProposed + 1;
        
    % Calculate Gradient
    mexpx          = m*exp(xNew);
    GradL          = y - mexpx - SigmaInv*(xNew - muOnes);
    
    % Calculate current log-joint-likelihood
    LVCurrentLJL     = y'*xNew - sum(mexpx) - 0.5*( (xNew - muOnes)'*SigmaInv*(xNew - muOnes) );
    
    % New tensor - constant metric given the parameters
    G              = SigmaInv;
    mexpMuSigma    = m*exp(muOnes + diag(Sigma));
    G(1:N^2+1:end) = mexpMuSigma' + G(1:N^2+1:end);
    CholG          = chol(G);
    InvG           = chol2inv(CholG);
    LogDetG        = 2*sum(log(diag(CholG)));
    CholInvG       = chol(InvG);
    
    
    ProposedMomentum = (randn(1,D)*CholG)';
    OriginalMomentum = ProposedMomentum;
        
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
        
    RandomSteps = ceil(rand*LVNumOfLeapFrogSteps);
        
    % Perform leapfrog steps
    for StepNum = 1:RandomSteps
            
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        % Note that partial derivative of metric tensor is zero
        ProposedMomentum = ProposedMomentum + (TimeStep*LVStepSize/2)*GradL;
        

        %%%%%%%%%%%%%%%%%%%%%%%
        % Update w parameters %
        %%%%%%%%%%%%%%%%%%%%%%%
        xNew = xNew + (TimeStep*LVStepSize)*(InvG*ProposedMomentum);
            
        
        % Calculate gradient based on new parameters
        mexpx = m*exp(xNew);
        GradL = y - mexpx - SigmaInv*(xNew - muOnes);
            
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        % Note that partial derivative of metric tensor is zero
        ProposedMomentum = ProposedMomentum + (TimeStep*LVStepSize/2)*GradL;
            
    end

    
    % Calculate proposed H value
    ProposedLJL    = y'*xNew - sum(mexpx) - 0.5*((xNew - muOnes)'*SigmaInv*(xNew - muOnes));
    ProposedH      = -ProposedLJL + (ProposedMomentum'*InvG*ProposedMomentum)/2;
        
    % Calculate current H value
    CurrentH      = -LVCurrentLJL + (OriginalMomentum'*InvG*OriginalMomentum)/2;
    
    % Accept according to ratio
    Ratio = -ProposedH + CurrentH;

    if Ratio > 0 || (Ratio > log(rand))
        LVCurrentLJL = ProposedLJL;
        x = xNew;
        LVAccepted = LVAccepted + 1;
        disp('Latent Variable Step Accepted')
        drawnow
    else
        disp('Latent Variable Step Rejected')
        drawnow
    end
    
    
        

    % Save samples if required
    if IterationNum > BurnIn
        LVSaved(IterationNum-BurnIn,:)  = x;
        HPSaved(IterationNum-BurnIn,:)  = [sigmaSq Beta];
        HPLJLSaved(IterationNum-BurnIn) = HPCurrentLJL;
        LVLJLSaved(IterationNum-BurnIn) = LVCurrentLJL;
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
save(['Results/RMHMC_Trans_Paras_LV_LogCox_' num2str(N) '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'HPSaved', 'LVSaved', 'HPLJLSaved', 'LVLJLSaved','TimeTaken', 'HPNumOfLeapFrogSteps', 'HPStepSize', 'LVNumOfLeapFrogSteps', 'LVStepSize')


function x = chol2inv(U)
    % Takes a cholesky decomposed matrix and returns the inverse of the
    % original matrix
    
    % This need the lightspeed toolbox!
    iU = inv_triu(U);
    x = iU*iU';
end


end


