function [ ] = BLR_RMHMC_StudentT( DataSet )

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
        %
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);

        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 6;
        StepSize           = 3/NumOfLeapFrogSteps;
        NumOfNewtonSteps   = 4;
        
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
        %
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);
        
        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        
        NumOfLeapFrogSteps = 6;
        StepSize           = 1/NumOfLeapFrogSteps;
        NumOfNewtonSteps   = 4;
        
        
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
        %
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);

        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);
        
        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 6;
        StepSize           = 3/NumOfLeapFrogSteps;
        NumOfNewtonSteps   = 4;

        
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
        %
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);
        
        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);

        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 6;
        StepSize           = 3/NumOfLeapFrogSteps;
        NumOfNewtonSteps   = 4;
        
        
        
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
        %
        XX = ones(size(X,1),1);
        for i = 1:Polynomial_Order
            XX = [XX X.^i];
        end

        [N,D] = size(XX);
        
        % Standardise Data
        X = (X-repmat(mean(X),N,1))./repmat(std(X),N,1);

        % HMC Setup
        NumOfIterations    = 6000;
        BurnIn             = 1000;
        NumOfLeapFrogSteps = 6;
        StepSize           = 3/NumOfLeapFrogSteps;
        NumOfNewtonSteps   = 4;

end


Proposed = 0;
Accepted = 0;

% Set initial values of w
w = ones(D,1)*1e-3;
wSaved = zeros(NumOfIterations-BurnIn,D);
warning off

% Pre-allocate memory for partial derivatives
for d = 1:D
    GDeriv{d} = zeros(D);
end

% Calculate joint log likelihood for current w
LogPrior      = LogNormPDF(zeros(1,D),w,alpha);
f             = XX*w;
LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
CurrentLJL    = LogLikelihood + LogPrior;


for IterationNum = 1:NumOfIterations
       
    %IterationNum
    
    if mod(IterationNum,50) == 0
        disp([num2str(IterationNum) ' iterations completed.'])
        disp(Accepted/Proposed)
        Proposed = 0;
        Accepted = 0;
        drawnow
    end
    
    wNew = w;
    Proposed = Proposed + 1;
    
    % Calculate G
    f = XX*wNew;
    p = 1./(1+exp(-f));
    v = p.*(ones(N,1)-p);
    G = (XX'.*repmat(v',D,1))*XX + (eye(D)./alpha);
    
    InvG = inv(G);
    
    OriginalG     = G;
    OriginalCholG = chol(G);
    OriginalInvG  = InvG;
    
    % Calculate the partial derivatives dG/dw
    % Faster to compute Z as vector then diag in the next calculation
    for d = 1:D
        Z = ((1 - 2*p).*XX(:,d));
        %GDeriv{d} = XX'*v*diag(Z)*XX; % Slow because of diag
        % faster
        Z1 = (v.*Z);
        for a =1:D Z2(:,a) = (XX(:,a).*Z1); end
        GDeriv{d} = Z2'*XX;
        
        InvGdG{d}      = InvG*GDeriv{d};
        TraceInvGdG(d) = trace(InvGdG{d});
        
    end
    
    try
        ProposedMomentum = mvtrnd(G, 1)';
    catch
        disp('error')
    end
    OriginalMomentum = ProposedMomentum;
    
    
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
    
    RandomSteps = ceil(rand*NumOfLeapFrogSteps);
    
    SavedSteps(1,:) = wNew;
    
    
    % Perform leapfrog steps
    for StepNum = 1:RandomSteps
        
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        
        dLdTheta  = ( XX'*( t - (1./(1+exp(-f))) ) - eye(D)*(1/alpha)*wNew );
        TraceTerm = 0.5*TraceInvGdG';
        
        % Multiple fixed point iteration
        PM = ProposedMomentum;
        for FixedIter = 1:NumOfNewtonSteps
            MomentumHist(FixedIter,:) = PM;
            
            InvGMomentum = InvG*PM;
            for d = 1:D
                LastTerm(d)  = ((1+D)/2)*(PM'*InvGdG{d}*InvGMomentum)/(1 + PM'*InvGMomentum);
            end
            
            PM = ProposedMomentum + TimeStep*(StepSize/2)*(dLdTheta - TraceTerm + LastTerm');
        end
        ProposedMomentum = PM;
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%
        % Update w parameters %
        %%%%%%%%%%%%%%%%%%%%%%%
        %%% Multiple Fixed Point Iteration %%%
        %
        OriginalInvGMomentum = G\ProposedMomentum;
        OriginalMomInvGMom   = ProposedMomentum'*OriginalInvGMomentum;
        
        Pw = wNew;
        for FixedIter = 1:NumOfNewtonSteps
            wHist(FixedIter,:) = Pw;
            
            f = XX*Pw;
            p = 1./(1+exp(-f));
            
            % faster
            v = p.*(ones(N,1)-p);
            G = (XX'.*repmat(v',D,1))*XX + (eye(D)./alpha);
            
            InvGMomentum = G\ProposedMomentum;
            
            Pw = wNew + (TimeStep*(StepSize/2)*(1+D))*OriginalInvGMomentum/(1 + OriginalMomInvGMom) + (TimeStep*(StepSize/2)*(1+D))*InvGMomentum/(1 + ProposedMomentum'*InvGMomentum);
        end
        wNew = Pw;
        
        
        % Calculate G based on new parameters
        f = XX*wNew;
        p = 1./(1+exp(-f));
        
        % faster
        v = p.*(ones(N,1)-p);
        %for a = 1:D v1(:,a) = (XX(:,a).*v); end;
        %G = v1'*XX + (eye(D)./alpha);
        G = (XX'.*repmat(v',D,1))*XX + (eye(D)./alpha);
        
        InvG = inv(G);
        
        
        % Calculate the partial derivatives dG/dw
        % Faster to compute Z as vector then diag in the next calculation
        for d = 1:D
            Z = ((1 - 2*p).*XX(:,d));
            %GDeriv{d} = XX'*v*diag(Z)*XX; % Slow because of diag
            % faster
            Z1 = (v.*Z);
            for a =1:D Z2(:,a) = (XX(:,a).*Z1); end
            GDeriv{d} = Z2'*XX;
            
            InvGdG{d}      = InvG*GDeriv{d};
            TraceInvGdG(d) = trace(InvGdG{d});
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%
        % Update momentum %
        %%%%%%%%%%%%%%%%%%%
        f = XX*wNew;
        dLdTheta  = ( XX'*( t - (1./(1+exp(-f))) ) - eye(D)*(1/alpha)*wNew );
        TraceTerm = 0.5*TraceInvGdG';
        
        % Calculate last term in dH/dTheta
        InvGMomentum = (InvG*ProposedMomentum);
        for d = 1:D
            LastTerm(d) = ((1+D)/2)*(PM'*InvGdG{d}*InvGMomentum)/(1 + PM'*InvGMomentum);
        end
        
        ProposedMomentum = ProposedMomentum + TimeStep*(StepSize/2)*(dLdTheta - TraceTerm + LastTerm');
        
        SavedSteps( StepNum + 1, : ) = wNew;
    end
    
    % Calculate proposed H value
    LogPrior      = LogNormPDF(zeros(1,D),wNew,alpha);
    f             = XX*wNew;
    LogLikelihood = f'*t - sum(log(1+exp(f))); %training likelihood
    ProposedLJL   = LogLikelihood + LogPrior;
    
    ProposedLogDet = 0.5*( 2*sum(log(diag(chol(G)))) );
    
    ProposedH = -ProposedLJL + ProposedLogDet + ((1+D)/2)*log(1 + ProposedMomentum'*InvG*ProposedMomentum);
    
    
    % Calculate current H value
    CurrentLogDet = 0.5*( 2*sum(log(diag(OriginalCholG))) );
    
    CurrentH  = -CurrentLJL + CurrentLogDet + ((1+D)/2)*log(1 + OriginalMomentum'*OriginalInvG*OriginalMomentum);
    
    % Accept according to ratio
    Ratio = -ProposedH + CurrentH;
    
    if Ratio > 0 || (Ratio > log(rand))
        CurrentLJL = ProposedLJL;
        w = wNew;
        Accepted = Accepted + 1;
        %disp('Accepted')
    else
        %disp('Rejected')
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
save(['Results/Results_RMHMC_StudentT_' num2str(NumOfLeapFrogSteps) '_' num2str(NumOfNewtonSteps) '_' DataSet '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'betaPosterior', 'TimeTaken')

% Plot paths and histograms
figure(5)
plot(wSaved);
figure(6)
for d = 1:D
    subplot(ceil(D/4),4,d)
    hist(wSaved(:,d))
end



end
