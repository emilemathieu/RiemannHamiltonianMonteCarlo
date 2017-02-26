function [] = LGC_mMALA_LV()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up path for lightspeed toolbox %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('/scratch/bc/Software/lightspeed'))
%addpath(genpath('/Applications/Matlab_Addons/lightspeed'))

randn('state', sum(clock) );
rand('twister', sum(clock) );

% Grid Size
N     = 64;

% Load data
% Y is the data
% X are the latent variables
Data=load('TestData64.mat');
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

StepSize           = 0.07;

Proposed = 0;
Accepted = 0;

% Set initial values of w
muOnes   = ones(D,1)*mu;

% Start from mu
x        = muOnes;

xSaved   = zeros(NumOfIterations-BurnIn, D);
LJLSaved = zeros(NumOfIterations-BurnIn, 1);


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
CholInvG      = zeros(N^2);

% New tensor
G              = SigmaInv;
mexpMuSigma    = m*exp(muOnes + diag(Sigma));
G(1:N^2+1:end) = mexpMuSigma' + G(1:N^2+1:end);
CholG          = chol(G);
InvG           = chol2inv(CholG);
LogDetG        = 2*sum(log(diag(CholG)));
CholInvG       = chol(InvG);


for IterationNum = 1:NumOfIterations
    
    if mod(IterationNum,50) == 0
        disp([num2str(IterationNum) ' iterations completed.'])
        disp(Accepted/Proposed)
        Proposed = 0;
        Accepted = 0;
        drawnow
    end
    
    %disp(['Iteration ' num2str(IterationNum)])
    
    xNew           = x;
    Proposed       = Proposed + 1;
        
    % Calculate G
    mexpx          = m*exp(xNew);
    GradL          = y - mexpx - SigmaInv*(x - muOnes);
    
    % Old mean
    Mean           = xNew + (StepSize/2)*(InvG*GradL);
    
    R              = randn(1,N^2);
    xNew           = Mean + ( ( R*CholInvG )*sqrt(StepSize) )';
    
    
    ProbNewGivenOld = -0.5*( log(det(StepSize))-LogDetG ) - (0.5/StepSize)*((Mean-xNew)'*G*(Mean-xNew));
    
    
    % Calculate New G
    mexpx = m*exp(xNew);
    GradL = y - mexpx - SigmaInv*(xNew - muOnes);
    
    % Old mean
    Mean            = xNew + (StepSize/2)*(InvG*GradL);
    ProbOldGivenNew = -0.5*( log(det(StepSize))-LogDetG ) - (0.5/StepSize)*((Mean-x)'*G*(Mean-x));
    ProposedLJL     = y'*xNew - sum(mexpx) - 0.5*( (xNew - muOnes)'*SigmaInv*(xNew - muOnes) );
    
    
    % Accept according to ratio
    Ratio = ProposedLJL + ProbOldGivenNew - CurrentLJL - ProbNewGivenOld;

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
save(['Results/mMALA_LV_LogCox_' num2str(N) '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'StepSize', 'xSaved', 'LJLSaved', 'TimeTaken')

    

function x = chol2inv(U)
    % Takes a cholesky decomposed matrix and returns the inverse of the
    % original matrix
    iU = inv_triu(U);
    x = iU*iU';
end


end


