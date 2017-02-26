function [ ] = BLR_IWLS( DataSet )

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


% Calculate W
f = XX*w;
p = 1./(1+exp(-f));
v = p.*(ones(N,1)-p);
W    = v;
InvW = 1./v;


% Equivalent to but quicker than: XX'*diag(CurrentW)*XX
CurrentCov = inv(XX'.*repmat(W',D,1)*XX + eye(D)./alpha);

% Calculate new sample observations and sample new mean
TempObserv  = f  + (t - p).*InvW;
CurrentMean = CurrentCov*( XX'*(W.*TempObserv) );

for IterationNum = 1:NumOfIterations
    
    % Sample new parameters
    wNew = CurrentMean + ( randn(1,D)*chol(CurrentCov) )';
    
    Proposed = Proposed + 1;
    
    LogPrior      = LogNormPDF(zeros(1,D),wNew,alpha);
    f             = XX*wNew;
    LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
    ProposedLJL   = LogLikelihood + LogPrior;
    
    ProbNewGivenOld = -sum(diag(log(chol(CurrentCov+eye(D)*1e-6))));
    ProbNewGivenOld = ProbNewGivenOld - 0.5*(CurrentMean-wNew)'/CurrentCov*(CurrentMean-wNew);
    
    % Calculate new W
    p = 1./(1+exp(-f));
    v = p.*(ones(N,1)-p);
    W    = v;
    InvW = 1./v;
    
    TempObserv    = f + (t - p).*InvW;
    
    % Calculate new mean and covariance
    NewCov = inv(XX'.*repmat(W',D,1)*XX + eye(D)./alpha);
    NewMean = NewCov*( XX'*(W.*TempObserv) );
    
    ProbOldGivenNew = -sum(diag(log(chol(NewCov+eye(D)*1e-6))));
    ProbOldGivenNew = ProbOldGivenNew - 0.5*(NewMean-w)'/NewCov*(NewMean-w);
    
    % Accept according to ratio
    Ratio = ProposedLJL + ProbOldGivenNew - CurrentLJL - ProbNewGivenOld;
    
    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        w = wNew;
        
        CurrentMean = NewMean;
        CurrentCov  = NewCov;
        
        Accepted   = Accepted + 1;
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
        
        AcceptanceRatio = Accepted/Proposed;
            
        % Print acceptance ratios
        disp(AcceptanceRatio)
        
        % Reset counters
        Accepted = 0;
        Proposed = 0;
    end

    % Start timer after burn-in
    if IterationNum == BurnIn
        disp('Burn-in complete, now drawing posterior samples.')
        drawnow
        %{
        disp('Acceptance ratios:')
        disp(AcceptanceRatio)
        %}
       
        tic;
    end
end

% Stop timer
TimeTaken = toc;

betaPosterior = wSaved;

CurTime = fix(clock); % Unique file identifier
cd('..')
save(['Results/Results_IWLS_' DataSet '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'betaPosterior', 'TimeTaken')

% Plot paths and histograms
figure(50)
plot(wSaved);
figure(51)
for d = 1:D
    subplot(ceil(D/4),4,d)
    hist(wSaved(:,d))
end



end
