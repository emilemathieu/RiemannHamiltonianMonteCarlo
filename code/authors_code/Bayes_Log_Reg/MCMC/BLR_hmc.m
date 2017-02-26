function [ ] = BLR_hmc( DataSet )

% Random Numbers...
randn('state', sum(100*clock));
rand('twister', sum(100*clock));

switch(DataSet)

    case 'Australian'
    
        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/australian.mat');
        t=X(:,end);
        X(:,end)=[];
        
        % Normalise Data
        [N, D] = size(X);
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);

        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end
        [N,D] = size(XX);

        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 100;
        
        StepSize           = 0.1;
        Mass               = diag(ones(D,1)*1);

        
    case 'German'

        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/german.mat');
        t=X(:,end);
        X(:,end)=[];

        % Normalise Data
        [N, D] = size(X);
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        % German Credit - replace all 1s in t with 0s
        t(find(t==1)) = 0;
        % German Credit - replace all 2s in t with 1s
        t(find(t==2)) = 1;

        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);
        
        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 100;
        StepSize           = 0.05;
        Mass               = diag(ones(D,1)*1);
        
        
    case 'Heart'
        
        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/heart.mat');
        t=X(:,end);
        X(:,end)=[];

        % Normalise Data
        [N, D] = size(X);
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        % German Credit - replace all 1s in t with 0s
        t(find(t==1)) = 0;
        % German Credit - replace all 2s in t with 1s
        t(find(t==2)) = 1;

        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);

        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 100;
        StepSize           = 0.14;
        Mass               = diag(ones(D,1)*1);
        
    case 'Pima'

        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/pima.mat');
        t=X(:,end);
        X(:,end)=[];

        % Normalise Data
        [N, D] = size(X);
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);

        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 100;
        StepSize           = 0.1;
        Mass               = diag(ones(D,1)*1);
        
    case 'Ripley'

        %Two hyperparameters of model
        Polynomial_Order = 3;
        alpha=100;

        %Load and prepare train & test data
        load('./Data/ripley.mat');
        t=X(:,end);
        X(:,end)=[];

        % Normalise Data
        [N, D] = size(X);
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);

        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 100;
        StepSize           = 0.14;
        Mass               = diag(ones(D,1)*1);
end

Proposed = 0;
Accepted = 0;

InvMass = sparse(inv(Mass));

% Set initial values of w
w = zeros(D,1);
wSaved = zeros(NumOfIterations-BurnIn,D);

% Calculate joint log likelihood for current w
LogPrior      = LogNormPDF(zeros(1,D),w,alpha);
f             = XX*w;
LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
CurrentLJL    = LogLikelihood + LogPrior;


for IterationNum = 1:NumOfIterations
        
    if mod(IterationNum,50) == 0
        disp(IterationNum)
    end
    
    % Sample momentum
    ProposedMomentum = (randn(1,D)*Mass)';
    OriginalMomentum = ProposedMomentum;
        
    wNew = w;
    Proposed = Proposed + 1;
      
    RandomStep = ceil(rand*NumOfLeapFrogSteps);
        
    % Perform leapfrog steps
    for StepNum = 1:RandomStep
        f = XX*wNew;
        ProposedMomentum = ProposedMomentum + (StepSize/2).*( XX'*( t - (exp(f)./(1+exp(f))) ) - eye(D)*(1/alpha)*wNew );
            
        if sum(isnan(ProposedMomentum)) > 0
            break
        end
        
        wNew = wNew + StepSize.*(ProposedMomentum);            
            
        f = XX*wNew;
        ProposedMomentum = ProposedMomentum + (StepSize/2).*( XX'*( t - (exp(f)./(1+exp(f))) ) - eye(D)*(1/alpha)*wNew );
            
    end
        
    % Calculate proposed H value
    LogPrior      = LogNormPDF(zeros(1,D),wNew,alpha);
    f             = XX*wNew;
    LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
    ProposedLJL   = LogLikelihood + LogPrior;
    
    ProposedH = -ProposedLJL + (ProposedMomentum'*InvMass*ProposedMomentum)/2;
        
    % Calculate current H value
    CurrentH  = -CurrentLJL + (OriginalMomentum'*InvMass*OriginalMomentum)/2;
       
    % Accept according to ratio
    Ratio = -ProposedH + CurrentH;
        
        
    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        w = wNew;
        Accepted = Accepted + 1;
    end

    % Save samples if required
    if IterationNum > BurnIn
        wSaved(IterationNum-BurnIn,:) = w;
    elseif mod(IterationNum,50) == 0
        disp([num2str(IterationNum) ' iterations completed.'])
        disp(['Acceptance: ' num2str(Accepted/Proposed)])
        Accepted = 0;
        Proposed = 0;
    end
    
    % Start timer after burn-in
    if IterationNum == BurnIn
        disp('Burn-in complete, now drawing posterior samples.')
        
        tic;
    end
    
end

% Stop timer
TimeTaken = toc;

betaPosterior = wSaved;

CurTime = fix(clock);
cd('..')
save(['Results/Results_HMC_' DataSet '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'betaPosterior', 'TimeTaken')


% Plot paths and histograms
figure(70)
plot(wSaved);
figure(71)
for d = 1:D
    subplot(ceil(D/4),4,d)
    hist(wSaved(:,d))
end


end
