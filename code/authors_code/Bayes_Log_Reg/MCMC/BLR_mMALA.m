function [ ] = BLR_mMALA( DataSet )

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
        NumOfIterations = 10000;
        BurnIn          = 5000;
        StepSize        = 1;
        
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
        NumOfIterations = 10000;
        BurnIn          = 5000;
        StepSize        = 1;
        
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
        

        % Langevin Setup
        NumOfIterations = 10000;
        BurnIn          = 5000;
        StepSize        = 1;
        
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
        NumOfIterations = 10000;
        BurnIn          = 5000;
        StepSize        = 1;
        
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
        NumOfIterations = 10000;
        BurnIn          = 5000;
        StepSize        = 1;
        
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


%%% Pre-Calculations %%%
wNew = w;

% Calculate G based on new parameters
f = XX*wNew;
p = 1./(1+exp(-f)); 
% faster
v = p.*(ones(N,1)-p);
for a = 1:D v1(:,a) = (XX(:,a).*v); end;
CurrentG = v1'*XX + (eye(D)./alpha);
    
% Inverse of G
CurrentInvG = inv(CurrentG);
    
CurrentFirstTerm = (CurrentInvG*( XX'*( t - (exp(f)./(1+exp(f))) ) - eye(D)*(1/alpha)*wNew ));
    
for d = 1:D
    Z = ((1 - 2*p).*XX(:,d));
    %GDeriv{d} = XX'*v*diag(Z)*XX; % Slow because of diag                
    % faster
    Z1 = (v.*Z);
    for a =1:D Z2(:,a) = (XX(:,a).*Z1); end
    GDeriv{d} = Z2'*XX;
                
    InvGdG{d}      = CurrentInvG*GDeriv{d};
    TraceInvGdG(d) = trace(InvGdG{d});
        
    CurrentSecondTerm(:,d) = InvGdG{d}*CurrentInvG(:,d);
end
    
CurrentThirdTerm = CurrentInvG*TraceInvGdG';




for IterationNum = 1:NumOfIterations
        
    if mod(IterationNum,1000) == 0  && IterationNum < BurnIn
        disp([num2str(IterationNum) ' iterations completed.'])
        drawnow
    end
        
    wNew     = w;
    Proposed = Proposed + 1;
    
    
    % Calculate the drift term
    Mean = wNew + (StepSize/(2))*CurrentFirstTerm...
                - StepSize*sum(CurrentSecondTerm,2)... % Sum across columns
                + (StepSize/(2))*CurrentThirdTerm;
    
    wNew = Mean + ( randn(1,D)*chol(StepSize*CurrentInvG) )';
    
    % Calculate proposed Likelihood value
    LogPrior      = LogNormPDF(zeros(1,D),wNew,alpha);
    f             = XX*wNew;
    LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
    ProposedLJL   = LogLikelihood + LogPrior;

    ProbNewGivenOld = -sum(log(diag(chol(StepSize*CurrentInvG)))) - 0.5*(Mean-wNew)'*(CurrentG/StepSize)*(Mean-wNew);
    
    
    %%% Calculate probability of Old given New %%%
    
    % Calculate G based on new parameters
    f = XX*wNew;
    p = 1./(1+exp(-f)); 
    % faster
    v = p.*(ones(N,1)-p);
    for a = 1:D v1(:,a) = (XX(:,a).*v); end;
    G = v1'*XX + (eye(D)./alpha);
    
    % Inverse of G
    InvG = inv(G);
    
    FirstTerm = (InvG*( XX'*( t - (exp(f)./(1+exp(f))) ) - eye(D)*(1/alpha)*wNew ));
    
    for d = 1:D
        Z = ((1 - 2*p).*XX(:,d));
        %GDeriv{d} = XX'*v*diag(Z)*XX; % Slow because of diag                
        % faster
        Z1 = (v.*Z);
        for a =1:D Z2(:,a) = (XX(:,a).*Z1); end
        GDeriv{d} = Z2'*XX;
                
        InvGdG{d}     = InvG*GDeriv{d};
        TraceInvGdG(d)   = trace(InvGdG{d});
        
        SecondTerm(:,d) = InvGdG{d}*InvG(:,d);
    end
    
    ThirdTerm = InvG*TraceInvGdG';
    
    
    % Calculate the drift term
    Mean = wNew + (StepSize/(2))*FirstTerm...
                - StepSize*sum(SecondTerm,2)... % Sum across columns
                + (StepSize/(2))*ThirdTerm;
    
    ProbOldGivenNew = -sum(log(diag(chol(StepSize*InvG)))) - 0.5*(Mean-w)'*(G/StepSize)*(Mean-w); 
        
    % Accept according to ratio
    Ratio = ProposedLJL + ProbOldGivenNew - CurrentLJL - ProbNewGivenOld;
        
        
    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        
        CurrentG          = G;
        CurrentInvG       = InvG;
        CurrentFirstTerm  = FirstTerm;
        CurrentSecondTerm = SecondTerm;
        CurrentThirdTerm  = ThirdTerm;
        
        w = wNew;
        Accepted = Accepted + 1;
        
        %disp('Accepted')
    else
        %disp('Rejected')
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
        tic;
    end
    
end

% Stop timer
TimeTaken = toc;

betaPosterior = wSaved;

CurTime = fix(clock);
cd('..')
save(['Results/Results_mMALA_' DataSet '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'betaPosterior', 'TimeTaken')


% Plot paths and histograms
figure(60)
plot(wSaved);
figure(61)
for d = 1:D
    subplot(ceil(D/4),4,d)
    hist(wSaved(:,d))
end



end
