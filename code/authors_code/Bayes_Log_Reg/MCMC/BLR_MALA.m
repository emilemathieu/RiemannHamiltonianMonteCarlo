function [ ] = BLR_MALA( DataSet )

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

        N = size(X,1);
        
        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end
        [N,D] = size(XX);

        % Langevin Setup
        NumOfIterations = 25000;
        BurnIn          = 20000;
        StepSize        = 0.04;
        Scaling         = D^(1/2);
        
    case 'German'

        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/german.mat');
        t=X(:,end);
        X(:,end)=[];

        N = size(X,1);
        
        % Standardise Data
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
        
        % Langevin Setup
        NumOfIterations = 25000;
        BurnIn          = 20000;
        StepSize        = 0.013;
        Scaling         = D^(1/2);
        
    case 'Heart'
        
        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/heart.mat');
        t=X(:,end);
        X(:,end)=[];

        N = size(X,1);
        
        % Standardise Data
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
        
        % Langevin Setup
        NumOfIterations = 25000;
        BurnIn          = 20000;
        StepSize        = 0.075;
        Scaling         = D^(1/2);
        
    case 'Pima'

        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/pima.mat');
        t=X(:,end);
        X(:,end)=[];
        
        N = size(X,1);
        
        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);

        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);
        
        % Langevin Setup
        NumOfIterations = 25000;
        BurnIn          = 20000;
        StepSize        = 0.025;
        Scaling         = D^(1/2);
        
        
    case 'Ripley'

        %Two hyperparameters of model
        Polynomial_Order = 3;
        alpha=100;

        %Load and prepare train & test data
        load('./Data/ripley.mat');
        t=X(:,end);
        X(:,end)=[];
        
        N = size(X,1);
        
        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);

        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);
        
        % Langevin Setup
        NumOfIterations = 25000;
        BurnIn          = 20000;
        StepSize        = 0.1;
        Scaling         = 2*D^(1/2);

end



Proposed = 0;
Accepted = 0;


% Set initial values of w
w = zeros(D,1);
wSaved = zeros(NumOfIterations-BurnIn,D);

% Calculate joint log likelihood for current w
LogPrior      = LogNormPDF(zeros(1,D),w,alpha);
f             = XX*w;
LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
CurrentLJL    = LogLikelihood + LogPrior;


for IterationNum = 1:NumOfIterations
        
    if mod(IterationNum,1000) == 0  && IterationNum < BurnIn
        disp([num2str(IterationNum) ' iterations completed.'])
        drawnow
    end
        
    wNew     = w;
    Proposed = Proposed + 1;
        
    f    = XX*wNew;
    Mean = wNew + (StepSize/(2*Scaling))*( XX'*( t - (exp(f)./(1+exp(f))) ) - eye(D)*(1/alpha)*wNew );
    wNew = Mean + randn(D,1)*sqrt((StepSize/Scaling));
            
    % Calculate proposed H value
    LogPrior      = LogNormPDF(zeros(1,D),wNew,alpha);
    f             = XX*wNew;
    LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
    ProposedLJL   = LogLikelihood + LogPrior;
        
    ProbNewGivenOld = LogNormPDF(Mean, wNew,(StepSize/Scaling));
        
        
    % Calculate probability of Old given New
    f    = XX*wNew;
    Mean = wNew + (StepSize/(2*Scaling))*( XX'*( t - (exp(f)./(1+exp(f))) ) - eye(D)*(1/alpha)*wNew );
    ProbOldGivenNew = LogNormPDF(w, Mean, (StepSize/Scaling));
        
        
    % Accept according to ratio
    Ratio = ProposedLJL + ProbOldGivenNew - CurrentLJL - ProbNewGivenOld;
        
        
    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        w = wNew;
        Accepted = Accepted + 1;
    end
        
    if mod(IterationNum, 100) == 1 && IterationNum < BurnIn
        Acceptance = Accepted/Proposed;
        disp(Acceptance)
        
        Proposed = 0;
        Accepted = 0;
    end
        
    % Save samples if required
    if IterationNum > BurnIn
        wSaved(IterationNum-BurnIn,:) = w;
    end
    
    % Start timer after burn-in
    if IterationNum == BurnIn
        disp('Burn-in complete, now drawing posterior samples.')
        Scaling = D^(1/3);
        tic;
    end
    
end

% Stop timer
TimeTaken = toc;

betaPosterior = wSaved;

CurTime = fix(clock);
cd('..')
save(['Results/Results_MALA_' DataSet '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'betaPosterior', 'TimeTaken')


% Plot paths and histograms
figure(60)
plot(wSaved);
figure(61)
for d = 1:D
    subplot(ceil(D/4),4,d)
    hist(wSaved(:,d))
end



end
