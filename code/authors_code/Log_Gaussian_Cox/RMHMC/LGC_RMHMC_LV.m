function [] = LGC_RMHMC_LV()

addpath(genpath('/scratch/bc/Software/lightspeed'))
%addpath(genpath('/Applications/Matlab_Addons/lightspeed'))

randn('state', str2num(getenv('JOB_ID')) );
rand('twister', str2num(getenv('JOB_ID')) );

disp(str2num(getenv('JOB_ID')))

% Grid Size
N     = 64;

% Load data
% Y is the data
% X are the latent variables
Data=load('TestData64_7.mat');
y = Data.Y;


% Hyperparameters of model
s     = 1.91;
b     = 1/33;
mu    = log(126) - s/2;
m     = 1/(N^2);

[D] = length(y);

% HMC Setup
NumOfIterations    = 6000;
BurnIn             = 1000;
NumOfLeapFrogSteps = 30;
StepSize           = 0.1;

Proposed = 0;
Accepted = 0;

% Set initial values of w
muOnes   = ones(D,1)*mu;

% Start from mu
x        = muOnes;
%x = Data.X;

% Start from near posterior mode - see (c) Pg.262 in RSS-B, Christensen et al.
%{
for d = 1:D
    yCurrent    = y(d);
    options=optimset('Display','off');
    [x(d),fval] = fsolve(@NearPosterior,mu,options);  % Call optimizer
end
%}

ySaved   = zeros(NumOfIterations-BurnIn, D);
LJLSaved = zeros(NumOfIterations-BurnIn, 1);


% Calculate covariance matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SigmaInv       = zeros(N^2);
Sigma          = zeros(N^2);
x_r            = 0:1/(N - 1):1;
y_r            = 0:1/(N - 1):1;
[xs, ys]       = meshgrid(x_r, y_r);
coord_xy(:, 1) = xs(:);
coord_xy(:, 2) = ys(:);

%%% FASTER %%%
for i=1:N^2,
    coords1    = repmat(coord_xy(i,:), N^2, 1) - coord_xy; 
    Sigma(i,:) = sum(coords1.^2, 2).^0.5;
end


% Free up memory
coords1  = [];
coord_xy = [];

Sigma    = s.*exp( -Sigma./(b*N) );

SigmaInv = chol2inv(chol(Sigma));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CurrentLJL =  y'*x - sum(m*exp(x)) - 0.5*(x - muOnes)'*SigmaInv*(x - muOnes);


% Set aside memory for G etc
G             = zeros(N^2);
CholG         = zeros(N^2);
InvG          = zeros(N^2);


% New constant tensor
G              = SigmaInv;
mexpMuSigma    = m*exp(muOnes + diag(Sigma));
G(1:N^2+1:end) = mexpMuSigma' + G(1:N^2+1:end);
CholG          = chol(G);
InvG           = chol2inv(CholG);
LogDetG        = 2*sum(log(diag(CholG)));


%OriginalCholG = zeros(N^2);
%OriginalInvG  = zeros(N^2);

OriginalCholG = CholG;
OriginalInvG  = InvG;

for IterationNum = 1:NumOfIterations
       
   
    if mod(IterationNum,50) == 0
        disp([num2str(IterationNum) ' iterations completed.'])
        disp(Accepted/Proposed)
        Proposed = 0;
        Accepted = 0;
        drawnow
    end
    
    %disp(['Iteration ' num2str(IterationNum)])
    
    xNew = x;
    Proposed = Proposed + 1;
        
    % Calculate Gradient
    mexpx = m*exp(xNew);
    GradL = y - mexpx - SigmaInv*(x - muOnes);
            
    %G              = SigmaInv;
    %G(1:N^2+1:end) = mexpy' + G(1:N^2+1:end); % Add to diagonal
        
    %OriginalCholG = chol(G);
    %InvG          = chol2inv(OriginalCholG);
    
    %OriginalInvG = InvG;
        
    % Calculate trace of invG * dGdy
    %TraceInvGdG = diag(InvG).*(mexpy);
    
    ProposedMomentum = (randn(1,D)*OriginalCholG)';
    OriginalMomentum = ProposedMomentum;
        
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
        
    RandomSteps = ceil(rand*NumOfLeapFrogSteps);
        
    % Perform leapfrog steps
    for StepNum = 1:RandomSteps
            
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        %LogDetTerm       = (0.5)*TraceInvGdG;
        LogDetTerm = 0;
        ProposedMomentum = ProposedMomentum + (TimeStep*StepSize/2)*( GradL - LogDetTerm );
        
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%       
        %ProposedMomentum = Delta_LGC( ProposedMomentum, TimeStep*(StepSize/2), InvG, mexpy );

        %%%%%%%%%%%%%%%%%%%%%%%
        % Update w parameters %
        %%%%%%%%%%%%%%%%%%%%%%%
        xNew = xNew + (TimeStep*StepSize)*(InvG*ProposedMomentum);
            
        
        % Calculate G based on new parameters
        mexpx = m*exp(xNew);

        %G              = SigmaInv;
        %G(1:N^2+1:end) = mexpy' + G(1:N^2+1:end);
        
        %CholG = chol(G);
        %InvG  = chol2inv(CholG);
            

        % Calculate trace
        %TraceInvGdG = diag(InvG).*(mexpy);
        
        
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        %ProposedMomentum = Delta_LGC(ProposedMomentum, TimeStep*(StepSize/2), InvG, mexpy);
            
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        GradL            = y - mexpx - SigmaInv*(xNew - muOnes);
        %LogDetTerm       = (0.5).*TraceInvGdG;
        LogDetTerm = 0;
        ProposedMomentum = ProposedMomentum + (TimeStep*StepSize/2)*( GradL - LogDetTerm );
            
    end

    
    % Calculate proposed H value
    ProposedLJL    = y'*xNew - sum(mexpx) - 0.5*((xNew - muOnes)'*SigmaInv*(xNew - muOnes));
    %ProposedLogDet = 0.5*( log(2) + D*log(pi) + 2*sum(log(diag(CholG))) );
    %ProposedH      = -ProposedLJL + ProposedLogDet + (ProposedMomentum'*InvG*ProposedMomentum)/2;
    ProposedH      = -ProposedLJL + (ProposedMomentum'*InvG*ProposedMomentum)/2;
        
    % Calculate current H value
    %CurrentLogDet = 0.5*( log(2) + D*log(pi) + 2*sum(log(diag(OriginalCholG))) );
    %CurrentH      = -CurrentLJL + CurrentLogDet + (OriginalMomentum'*OriginalInvG*OriginalMomentum)/2;
    CurrentH      = -CurrentLJL + (OriginalMomentum'*OriginalInvG*OriginalMomentum)/2;
    
    % Accept according to ratio
    Ratio = -ProposedH + CurrentH;

    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        x = xNew;
        Accepted = Accepted + 1;
        disp('Accepted')
        drawnow
    else
        disp('Rejected')
        drawnow
    end
        

    % Save samples if required
    if IterationNum > BurnIn
        xSaved(IterationNum-BurnIn,:) = x;
        LJLSaved(IterationNum-BurnIn) = CurrentLJL;
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
save(['Results/RMHMC_LV_LogCox_' num2str(N) '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'NumOfLeapFrogSteps', 'StepSize', 'xSaved', 'LJLSaved', 'TimeTaken')

    
function F = NearPosterior(xTemp)

F = [yCurrent - exp(xTemp) - (xTemp-b)./s];

end

end


