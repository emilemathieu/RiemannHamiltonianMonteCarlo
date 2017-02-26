function [] = LGC_MALA_Transient()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up path for lightspeed toolbox %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('/scratch/bc/Software/lightspeed'))
%addpath(genpath('C:\Program Files\Matlab Addons\lightspeed'))

randn('state', sum(clock) );
rand('twister', sum(clock) );

% Grid Size
N     = 64;

% Y is the data
% X is the latent field

% Load data
Data=load('TestData64.mat');
y = Data.Y;


% Hyperparameters of model
s     = 1.91;
b     = 1/33;
mu    = log(126) - s/2;
m     = 1/(N^2);



[D]      = length(y);
StepSize = sqrt(2)^2; % optimised for transition phase
Scaling  = (N^2)^(1/2);


NumOfIterations    = 11000;
BurnIn             = 6000;


Proposed = 0;
Accepted = 0;

% Set initial values of y
muOnes   = ones(D,1)*mu;
x        = muOnes;


ySaved   = zeros(NumOfIterations-BurnIn,D);
LJLSaved = zeros(1,NumOfIterations-BurnIn);
f_x_ySaved = zeros(1,NumOfIterations-BurnIn);


% Calculate covariance matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SigmaInv       = zeros(N^2);
x_r            = 0:1/(N - 1):1;
y_r            = 0:1/(N - 1):1;
[xs, ys]       = meshgrid(x_r, y_r);
coord_xy(:, 1) = xs(:);
coord_xy(:, 2) = ys(:);

%%% FASTER %%%
for i=1:N^2,
    coords1       = repmat(coord_xy(i,:), N^2, 1) - coord_xy; 
    SigmaInv(i,:) = sum(coords1.^2, 2).^0.5;
end
SigmaInv = s.*exp( -SigmaInv./(b*N) );

SigmaInv = chol2inv(chol(SigmaInv));

% Free up memory
coords1  = [];
coord_xy = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Transform y to Gamma
Omega = chol2inv( chol(SigmaInv + diag(x)) );
L     = chol(Omega);
Gamma = ((x - muOnes)'/L)';


CurrentLJL =  y'*x - sum(m*exp(x)) - 0.5*Gamma'*L'*SigmaInv*L*Gamma;

f_x_y =  y'*x - sum(m*exp(x)) - 0.5*(x - muOnes)'*SigmaInv*(x - muOnes);

for IterationNum = 1:NumOfIterations
    
    disp(['Iteration ' num2str(IterationNum)])
    
    xNew     = x;
    GammaNew = Gamma;
    LNew     = L;
    Proposed = Proposed + 1;
    
    
    mexpx = m*exp(xNew);
    GradL = LNew'*(y - mexpx) - LNew'*(SigmaInv*(LNew*GammaNew));
    Mean  = GammaNew + GradL*(StepSize/(Scaling*2));
    
    %%%%%%%%%%%%%%%%%%%%%
    % Update parameters %
    %%%%%%%%%%%%%%%%%%%%%
    GammaNew = Mean + randn(D,1)*sqrt((StepSize/Scaling));            
    xNew     = muOnes + L*GammaNew;
    Omega    = chol2inv( chol( SigmaInv + diag(xNew) ) );
    LNew     = chol(Omega);
    
    ProbNewGivenOld = LogNormPDF(Mean, GammaNew, (StepSize/Scaling));
        
    mexpx           = m*exp(xNew);
    GradL           = LNew'*(y - mexpx) - LNew'*(SigmaInv*(LNew*GammaNew));
    Mean            = GammaNew + GradL*(StepSize/(Scaling*2));
    ProbOldGivenNew = LogNormPDF(Gamma, Mean, (StepSize/Scaling));
    
    
    % Calculate proposed likelihood
    ProposedLJL    = y'*xNew - sum(mexpx) - 0.5*GammaNew'*LNew'*SigmaInv*LNew*GammaNew;
        
    
    % Accept according to ratio
    Ratio = ProposedLJL + ProbOldGivenNew - CurrentLJL - ProbNewGivenOld;
    
    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        Gamma      = GammaNew;
        x          = xNew;
        L          = LNew;
        Accepted = Accepted + 1;
        disp(CurrentLJL)
        disp('Accepted')
        drawnow
        
        f_x_y =  y'*xNew - sum(mexpx) - 0.5*(xNew - muOnes)'*SigmaInv*(xNew - muOnes);
    else
        disp('Rejected')
        drawnow
    end
     
    if mod(IterationNum,50) == 0
        disp([num2str(IterationNum) ' iterations completed.'])
        disp(Accepted/Proposed)
        Proposed = 0;
        Accepted = 0;
    end
    
    % Save samples if required
    if IterationNum > BurnIn
        xSaved(IterationNum-BurnIn,:) = x;
        LJLSaved(IterationNum-BurnIn) = CurrentLJL;
        
        f_x_ySaved(IterationNum-BurnIn) = f_x_y;
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
save(['Results/Results_MALA_Transient_LogCox_' num2str(N) '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'xSaved', 'LJLSaved', 'TimeTaken', 'f_x_ySaved')



function x = chol2inv(U)
    % Takes a cholesky decomposed matrix and returns the inverse of the
    % original matrix
    iU = inv_triu(U);
    x = iU*iU';
end



end
