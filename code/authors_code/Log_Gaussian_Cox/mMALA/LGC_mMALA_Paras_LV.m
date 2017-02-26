function [] = LGC_mMALA_Paras_LV()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up path for lightspeed toolbox %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('/scratch/bc/Software/lightspeed'))
%addpath(genpath('/Applications/Matlab_Addons/lightspeed'))

% Randomize
randn('state', sum(100*clock));
rand('twister', sum(100*clock));


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

% MALA Setup
NumOfIterations    = 6000;
BurnIn             = 1000;

HPStepSize         = 0.2;
LVStepSize         = 0.07;

% Counters
HPProposed = 0;
HPAccepted = 0;
LVProposed = 0;
LVAccepted = 0;

% Set initial values of w
muOnes   = ones(D,1)*Mu;

% Start from mu
x        = muOnes;


LVSaved     = zeros(NumOfIterations-BurnIn, D);
HPSaved     = zeros(NumOfIterations-BurnIn, NumOfHyperparameters);
LVLJLSaved  = zeros(NumOfIterations-BurnIn, 1);
HPLJLSaved  = zeros(NumOfIterations-BurnIn, 1);

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

SigmaChol = chol(Sigma);
SigmaInv = chol2inv(SigmaChol);

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

% Calculate G - Non-optimised version
%{
G1 = [];
G1(1,1) = 0.5*trace(SigmaInv*dSdsigmaSqT*SigmaInv*dSdsigmaSqT); %0.5*trace(SInvdSdsigmaSqAllSq);
G1(1,2) = 0.5*trace(SigmaInv*dSdsigmaSqT*SigmaInv*dSdBetaT); %0.5*sum(sum( SInvdSdsigmaSq.*SInvdSdBeta' ));
G1(2,1) = G1(1,2);
G1(2,2) = 0.5*trace(SigmaInv*dSdBetaT*SigmaInv*dSdBetaT); %0.5*trace(SInvdSdBetaAllSq);
%}   

% Calculate G - Optimised version
G = [];
G(1,1) = 0.5*sum(sum(SInvdSdsigmaSqT.*SInvdSdsigmaSqT'));
G(1,2) = 0.5*sum(sum(SInvdSdsigmaSqT.*SInvdSdBetaT'));
G(2,1) = G(1,2);
G(2,2) = 0.5*sum(sum(SInvdSdBetaT.*SInvdSdBetaT'));
% Subtract the second partial derivatives of prior
G(1,1) = G(1,1) - (-sigmaSq/GammaTheta);
G(2,2) = G(2,2) - (-Beta/GammaTheta);
    

OriginalCholG = chol(G);
InvG          = chol2inv(OriginalCholG);
    
LogDetG        = 2*sum(log(diag(OriginalCholG)));


%%% Calculate partial derivatives of metric tensor %%%
GDeriv{1}(1,1) = -sum(sum( SInvdSdsigmaSqTAllSq.*SInvdSdsigmaSqT' )) + trace( SInvdSdsigmaSqTAllSq );
GDeriv{1}(1,2) = -sum(sum( SInvdSdsigmaSqTAllSq.*SInvdSdBetaT' )) + sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaT' ));
GDeriv{1}(2,1) = GDeriv{1}(1,2);
GDeriv{1}(2,2) = -sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaTAllSq' )) + trace( SInvdSdBetaTAllSq );
GDeriv{1}(1,1) = GDeriv{1}(1,1) + sigmaSq/GammaTheta;

GDeriv{2}(1,1) = -sum(sum( SInvdSdBetaT.*SInvdSdsigmaSqTAllSq' )) + sum(sum( SInvdSdBetaT.*SInvdSdsigmaSqT' ));
GDeriv{2}(1,2) = -sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaTAllSq' )) + 0.5*trace( SInvdSdBetaTAllSq ) + 0.5*sum(sum( SInvdSdsigmaSqT.*SInvd2SdBetaT2' ));
GDeriv{2}(2,1) = GDeriv{2}(1,2);
GDeriv{2}(2,2) = -sum(sum( SInvdSdBetaTAllSq.*SInvdSdBetaT' )) + sum(sum( SInvd2SdBetaT2.*SInvdSdBetaT' ));
GDeriv{2}(2,2) = GDeriv{2}(2,2) + Beta/GammaTheta;


for d = 1:NumOfHyperparameters
    InvGdG{d}      = InvG*GDeriv{d};
    TraceInvGdG(d) = trace(InvGdG{d});
    
    SecondTerm(:,d) = InvGdG{d}*InvG(:,d);    
end




for IterationNum = 1:NumOfIterations
    
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
    
    disp(['Iteration ' num2str(IterationNum)])
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample the Hyperparameters %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    HPCurrentLJL = -0.5*( log(2) + NumOfHyperparameters*log(pi) + 2*sum(log(diag(SigmaChol))) ) - 0.5*(x - muOnes)'*SigmaInv*(x - muOnes);
    HPCurrentLJL = HPCurrentLJL + (Gammak-1)*log(sigmaSq) - sigmaSq/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for sigmaSq
    HPCurrentLJL = HPCurrentLJL + (Gammak-1)*log(Beta) - Beta/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for Beta
    HPCurrentJ   = log(sigmaSq*Beta); % Jacobian for transfomation
    
    HPNew          = [log(sigmaSq) log(Beta)];
    HPProposed     = HPProposed + 1;
    
    
    % Caluclate first and second partial derivatives of covariance function
    dSdsigmaSqT        = sigmaSq*exp(-Dist./(N*Beta));
    dSdBetaT           = Dist./(N*Beta).*dSdsigmaSqT;
    
    % Calculate Gradient
    TempVec = (x - muOnes)'*SigmaInv;
    
    % Gradient - Non-optimised
    %GradL(1) = -0.5*trace( SigmaInv*dSdsigmaSq ) + 0.5*(x - muOnes)'*SigmaInv*dSdsigmaSq*SigmaInv*(x - muOnes);
    %GradL(2) = -0.5*trace( SigmaInv*dSdBeta ) + 0.5*(x - muOnes)'*SigmaInv*dSdBeta*SigmaInv*(x - muOnes);
    
    % Optimised
    GradL = [];
    GradL(1) = -0.5*sum(sum( SigmaInv.*dSdsigmaSqT' )) + 0.5*TempVec*dSdsigmaSqT*TempVec';
    GradL(1) = GradL(1) + (Gammak-1) - sigmaSq/GammaTheta; % Add derivative of prior wrt parameter sigmaSq
    GradL(2) = -0.5*sum(sum( SigmaInv.*dSdBetaT' )) + 0.5*(x - muOnes)'*SigmaInv*dSdBetaT*SigmaInv*(x - muOnes);
    GradL(2) = GradL(2) + (Gammak-1) - Beta/GammaTheta; % Add derivative of prior wrt parameter Beta
    
    
    % Old mean
    Mean            = HPNew + (HPStepSize/2)*(InvG*GradL')'...
                        - HPStepSize*sum(SecondTerm,2)'... % Sum across columns
                        + (HPStepSize/(2))*(InvG*TraceInvGdG')';
    
    R               = randn(1,NumOfHyperparameters);
    HPNew           = Mean + ( ( R*chol(InvG) )*sqrt(HPStepSize) );
    
    ProbNewGivenOld = -0.5*( log(det(HPStepSize))-LogDetG ) - (0.5/HPStepSize)*((Mean-HPNew)*G*(Mean-HPNew)');
    
    
    % Calculate New G
    sigmaSqNew = exp(HPNew(1));
    BetaNew    = exp(HPNew(2));
        
    try
        
        SigmaNew     = sigmaSqNew.*exp( -Dist./(BetaNew*N) );
        SigmaCholNew = chol(SigmaNew+eye(N^2)*1e-6);
        SigmaInvNew  = chol2inv(SigmaCholNew);

        % Caluclate first and second partial derivatives of covariance function
        dSdsigmaSqT        = sigmaSqNew*exp(-Dist./(N*BetaNew));
        dSdBetaT           = Dist./(N*BetaNew).*dSdsigmaSqT;
        d2SdBetaT2         = -dSdBetaT + Dist./(N*BetaNew).*dSdBetaT;
        
        % Pre-calculate useful expressions
        SInvdSdsigmaSqT        = SigmaInv*dSdsigmaSqT;
        SInvdSdBetaT           = SigmaInv*dSdBetaT;
        SInvdSdsigmaSqTAllSq   = SInvdSdsigmaSqT*SInvdSdsigmaSqT;
        SInvdSdBetaTAllSq      = SInvdSdBetaT*SInvdSdBetaT;
        SInvd2SdBetaT2         = SigmaInv*d2SdBetaT2;

        % Calculate Gradient
        TempVec = (x - muOnes)'*SigmaInvNew;
        
        GradLNew = [];
        GradLNew(1) = -0.5*sum(sum( SigmaInvNew.*dSdsigmaSqT' )) + 0.5*TempVec*dSdsigmaSqT*TempVec';
        GradLNew(1) = GradLNew(1) + (Gammak-1) - sigmaSqNew/GammaTheta; % Add derivative of prior wrt parameter sigmaSq
        GradLNew(2) = -0.5*sum(sum( SigmaInvNew.*dSdBetaT' )) + 0.5*TempVec*dSdBetaT*TempVec';
        GradLNew(2) = GradLNew(2) + (Gammak-1) - BetaNew/GammaTheta; % Add derivative of prior wrt parameter Beta
        
        
        % Calculate G
        GNew = [];
        GNew(1,1)  = 0.5*sum(sum(SInvdSdsigmaSqT.*SInvdSdsigmaSqT')); %0.5*trace(SInvdSdsigmaSqAllSq);
        GNew(1,2)  = 0.5*sum(sum(SInvdSdsigmaSqT.*SInvdSdBetaT')); %0.5*sum(sum( SInvdSdsigmaSq.*SInvdSdBeta' ));
        GNew(2,1)  = GNew(1,2);
        GNew(2,2)  = 0.5*sum(sum(SInvdSdBetaT.*SInvdSdBetaT')); %0.5*trace(SInvdSdBetaAllSq);
        % Subtract negative second derivative
        GNew(1,1) = GNew(1,1) - (-sigmaSqNew/GammaTheta);
        GNew(2,2) = GNew(2,2) - (-BetaNew/GammaTheta);

        CholGNew  = chol(GNew);
        InvGNew   = chol2inv(CholGNew);

        LogDetGNew = 2*sum(log(diag(CholGNew)));
        
        %%% Calculate partial derivatives of metric tensor %%%
        GDeriv{1}(1,1) = -sum(sum( SInvdSdsigmaSqTAllSq.*SInvdSdsigmaSqT' )) + trace( SInvdSdsigmaSqTAllSq );
        GDeriv{1}(1,2) = -sum(sum( SInvdSdsigmaSqTAllSq.*SInvdSdBetaT' )) + sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaT' ));
        GDeriv{1}(2,1) = GDeriv{1}(1,2);
        GDeriv{1}(2,2) = -sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaTAllSq' )) + trace( SInvdSdBetaTAllSq );
        GDeriv{1}(1,1) = GDeriv{1}(1,1) + sigmaSqNew/GammaTheta;
        
        GDeriv{2}(1,1) = -sum(sum( SInvdSdBetaT.*SInvdSdsigmaSqTAllSq' )) + sum(sum( SInvdSdBetaT.*SInvdSdsigmaSqT' ));
        GDeriv{2}(1,2) = -sum(sum( SInvdSdsigmaSqT.*SInvdSdBetaTAllSq' )) + 0.5*trace( SInvdSdBetaTAllSq ) + 0.5*sum(sum( SInvdSdsigmaSqT.*SInvd2SdBetaT2' ));
        GDeriv{2}(2,1) = GDeriv{2}(1,2);
        GDeriv{2}(2,2) = -sum(sum( SInvdSdBetaTAllSq.*SInvdSdBetaT' )) + sum(sum( SInvd2SdBetaT2.*SInvdSdBetaT' ));
        GDeriv{2}(2,2) = GDeriv{2}(2,2) + BetaNew/GammaTheta;

        for d = 1:NumOfHyperparameters
            InvGdG{d}          = InvGNew*GDeriv{d};
            TraceInvGdGNew(d)  = trace(InvGdG{d});

            SecondTermNew(:,d) = InvGdG{d}*InvGNew(:,d);    
        end
        
        
        % Old mean
        Mean            = HPNew + (HPStepSize/2)*(InvGNew*GradLNew')'...
                            - HPStepSize*sum(SecondTermNew,2)'... % Sum across columns
                            + (HPStepSize/(2))*(InvGNew*TraceInvGdGNew')';

        ProbOldGivenNew = -0.5*( log(det(HPStepSize))-LogDetGNew ) - (0.5/HPStepSize)*((Mean-[log(sigmaSq) log(Beta)])*GNew*(Mean-[log(sigmaSq) log(Beta)])');

        ProposedLJL     = - 0.5*( log(2) + NumOfHyperparameters*log(pi) + 2*sum(log(diag(SigmaCholNew))) ) - 0.5*(x - muOnes)'*SigmaInvNew*(x - muOnes);
        ProposedLJL     = ProposedLJL + (Gammak-1)*log(sigmaSqNew) - sigmaSqNew/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for sigmaSq
        ProposedLJL     = ProposedLJL + (Gammak-1)*log(BetaNew) - BetaNew/GammaTheta - Gammak*log(GammaTheta) - log(gamma(Gammak)); % Prior for Beta
        HPProposedJ     = log(sigmaSqNew*BetaNew); % Jacobian

        % Accept according to ratio
        Ratio = ProposedLJL + HPProposedJ + ProbOldGivenNew - HPCurrentLJL - HPCurrentJ - ProbNewGivenOld;

        if Ratio > 0 || (Ratio > log(rand))
            HPCurrentLJL = ProposedLJL;
            sigmaSq      = exp(HPNew(1));
            Beta         = exp(HPNew(2));

            G       = GNew;
            InvG    = InvGNew;
            LogDetG = LogDetGNew;
            
            SecondTerm  = SecondTermNew;
            TraceInvGdG = TraceInvGdGNew;
            
            Sigma      = SigmaNew;
            SigmaChol  = SigmaCholNew;
            SigmaInv   = SigmaInvNew;

            HPAccepted = HPAccepted + 1;
            disp('HP Accepted')
            exp(HPNew)
            drawnow
        else
            disp('HP Rejected')
            Ratio
            drawnow
        end

    catch
        disp('HP Rejected')
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample the latent variables %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    xNew           = x;
    LVProposed     = LVProposed + 1;
        
    % Calculate Gradient
    mexpx          = m*exp(xNew);
    LVGradL        = y - mexpx - SigmaInv*(x - muOnes);
    
    % Calculate current log-joint-likelihood
    LVCurrentLJL     = y'*xNew - sum(mexpx) - 0.5*( (xNew - muOnes)'*SigmaInv*(xNew - muOnes) );
    
    % New tensor - constant given the parameters
    LVG              = SigmaInv;
    mexpMuSigma      = m*exp(muOnes + diag(Sigma));
    LVG(1:N^2+1:end) = mexpMuSigma' + LVG(1:N^2+1:end);
    CholLVG          = chol(LVG);
    InvLVG           = chol2inv(CholLVG);
    LogDetLVG        = 2*sum(log(diag(CholLVG)));
    CholInvLVG       = chol(InvLVG);
    
    
    % Old mean
    Mean            = xNew + (LVStepSize/2)*(InvLVG*LVGradL);
    
    R               = randn(1,N^2);
    xNew            = Mean + ( ( R*CholInvLVG )*sqrt(LVStepSize) )';
    
    ProbNewGivenOld = -0.5*( log(det(LVStepSize))-LogDetLVG ) - (0.5/LVStepSize)*((Mean-xNew)'*LVG*(Mean-xNew));
    
    
    % Calculate New G
    mexpx   = m*exp(xNew);
    LVGradL = y - mexpx - SigmaInv*(xNew - muOnes);
    
    % Old mean
    Mean            = xNew + (LVStepSize/2)*(InvLVG*LVGradL);
    
    ProbOldGivenNew = -0.5*( log(det(LVStepSize))-LogDetLVG ) - (0.5/LVStepSize)*((Mean-x)'*LVG*(Mean-x));
    ProposedLJL     = y'*xNew - sum(mexpx) - 0.5*( (xNew - muOnes)'*SigmaInv*(xNew - muOnes) );
    
    
    % Accept according to ratio
    Ratio = ProposedLJL + ProbOldGivenNew - LVCurrentLJL - ProbNewGivenOld;

    if Ratio > 0 || (Ratio > log(rand))
        LVCurrentLJL = ProposedLJL;
        x = xNew;
        LVAccepted = LVAccepted + 1;
        disp('LV Accepted')
        drawnow
    else
        disp('LV Rejected')
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
save(['Results/mMALA_Trans_Paras_LV_LogCox_' num2str(N) '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'HPStepSize', 'LVStepSize', 'HPSaved', 'LVSaved', 'HPLJLSaved', 'LVLJLSaved', 'TimeTaken')


function x = chol2inv(U)
    % Takes a cholesky decomposed matrix and returns the inverse of the
    % original matrix
    
    % This needs the lightspeed toolbox!
    iU = inv_triu(U);
    x = iU*iU';
end



end


