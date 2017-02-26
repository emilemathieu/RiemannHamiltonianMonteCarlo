function [ ] = BLR_metropolis( DataSet )

% Random Numbers...
randn('state', sum(100*clock));
rand('twister', sum(100*clock));

switch(DataSet)

    case 'Australian'
    
        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha = 100; % Variance of prior

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

        % Metropolis Setup
        NumOfIterations = 10000; % Total Number
        BurnIn          = 5000;
        ProposalSD      = ones(D,1);
        
        
    case 'German'

        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/german.mat');
        t=X(:,end);
        X(:,end)=[];

        % German Credit - replace all 1s in t with 0s
        t(find(t==1)) = 0;
        % German Credit - replace all 2s in t with 1s
        t(find(t==2)) = 1;

        N = size(X,1);
        
        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);
        
        % Metropolis Setup
        NumOfIterations = 10000; % Total Number
        BurnIn          = 5000;
        ProposalSD      = ones(D,1);
        
    case 'Heart'
        
        %Two hyperparameters of model
        Polynomial_Order = 1;
        alpha=100; 

        %Load and prepare train & test data
        load('./Data/heart.mat');
        t=X(:,end);
        X(:,end)=[];

        % German Credit - replace all 1s in t with 0s
        t(find(t==1)) = 0;
        % German Credit - replace all 2s in t with 1s
        t(find(t==2)) = 1;

        N = size(X,1);
        
        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        %Create Polynomial Basis
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);
        
        % Metropolis Setup
        NumOfIterations = 10000; % Total Number
        BurnIn          = 5000;
        ProposalSD      = ones(D,1);

        
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
        
        % Metropolis Setup
        NumOfIterations = 10000; % Total Number
        BurnIn          = 5000;
        ProposalSD      = ones(D,1);
        
        
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

        % Metropolis Setup
        NumOfIterations = 10000; % Total Number
        BurnIn          = 5000;
        ProposalSD      = ones(D,1);

end




Proposed = zeros(D,1);
Accepted = zeros(D,1);


% Set initial values of w
w = zeros(D,1);
wSaved = zeros(NumOfIterations-BurnIn,D);

% Calculate joint log likelihood for current w
LogPrior      = LogNormPDF(zeros(1,D),w,alpha);
f             = XX*w;
LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
CurrentLJL    = LogLikelihood + LogPrior;


for IterationNum = 1:NumOfIterations
    
    %IterationNum
    
    % For each w do metropolis step
    for d = 1:D
        wNew = w;
        wNew(d) = wNew(d) + randn*ProposalSD(d,1);
    
        Proposed(d) = Proposed(d) + 1;
        
        LogPrior      = LogNormPDF(zeros(1,D),wNew,alpha);
        f             = XX*wNew;
        LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
        ProposedLJL   = LogLikelihood + LogPrior;
    
        % Accept according to ratio
        Ratio = ProposedLJL - CurrentLJL;
        
        if Ratio > 0 || (Ratio > log(rand))
            CurrentLJL = ProposedLJL;
            w = wNew;
            Accepted(d) = Accepted(d) + 1;
        end
        
    end

    % Save samples if required
    if IterationNum > BurnIn
        wSaved(IterationNum-BurnIn,:) = w;
    end
    
    % Adjust sd every so often
    if mod(IterationNum,100) == 0 && IterationNum < BurnIn
        
        if mod(IterationNum,1000) == 0
            disp([num2str(IterationNum) ' iterations completed.'])
        end
        
        for d = 1:D
            AcceptanceRatio(d) = Accepted(d)/Proposed(d);
            
            if AcceptanceRatio(d) > 0.5
                ProposalSD(d) = ProposalSD(d)*1.2;
            elseif AcceptanceRatio(d) < 0.2
                ProposalSD(d) = ProposalSD(d)*0.8;
            end
            
        end
        
        % Print acceptance ratios
        %disp(AcceptanceRatio)
        
        % Reset counters
        Accepted = zeros(D,1);
        Proposed = zeros(D,1);
    end

    % Start timer after burn-in
    if IterationNum == BurnIn
        disp('Burn-in complete, now drawing posterior samples.')
        drawnow
        tic;
    end
end

% Stop timer
TimeTaken = toc;

betaPosterior = wSaved;

CurTime = fix(clock); % Unique file identifier
cd('..')
save(['Results/Results_Metropolis_' DataSet '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'betaPosterior', 'TimeTaken')

% Plot paths and histograms
figure(50)
plot(wSaved);
figure(51)
NumOfPlots = min(16,D);
for d = 1:NumOfPlots
    subplot(ceil(NumOfPlots/4),4,d)
    hist(wSaved(:,d))
end



end
