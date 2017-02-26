function [ ] = BLR_holmes_joint_update( DataSet )

% Random Numbers...
randn('state', sum(100*clock));
rand('twister', sum(100*clock));

switch(DataSet)
    
    case 'Australian'
    
        %Two hyperparameters of model
        Polynomial_Order = 1;
        v=100; 

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
        
        
    case 'German'

        %Two hyperparameters of model
        Polynomial_Order = 1;
        v=100; 

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
        
        
    case 'Heart'
        
        %Two hyperparameters of model
        Polynomial_Order = 1;
        v=100; 

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

    case 'Pima'

        %Two hyperparameters of model
        Polynomial_Order = 1;
        v=100; 

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
        
        
    case 'Ripley'

        %Two hyperparameters of model
        Polynomial_Order = 3;
        v=100; 

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
        
end



% Sampler Setup
NumOfIterations = 10000;
BurnIn          = 5000;

betaSaved = zeros(NumOfIterations-BurnIn,D);


vInv = eye(D)/v;


% Initialize mixing weights to vector of ones
Lambda = ones(N,1);

% Draw Z from truncated normal using randraw routine
Ones      = find(t==1);
MinusOnes = find(t==-1);
Z = zeros(N,1);

for a = Ones'
    Z(a) = rand_nort(0,1,0,inf);
end
for a = MinusOnes'
    Z(a) = rand_nort(0,1,-inf,0);
end

for i = 1:NumOfIterations

    if mod(i,100) == 0
        disp([num2str(i) ' iterations completed.'])
    elseif i == BurnIn
        disp('Burn-in complete, now drawing posterior samples.')
    end
    
    % Start timer after burn-in
    if i == BurnIn
        tic;
    end
    
    V = inv(XX'*diag(Lambda.^-1)*XX + vInv);
    L = chol(V,'lower');

    S = V*XX';
    B = S*diag(Lambda.^-1)*Z;

    for j = 1:N
        z_old = Z(j);
        H(j)  = XX(j,:)*S(:,j);
        W(j)  = H(j)/(Lambda(j)-H(j));
        m     = XX(j,:)*B;
        m     = m - W(j)*(Z(j)-m);
        q     = Lambda(j)*(W(j)+1);

        % Draw Z(j) from truncated normal
        if t(j) == 1
            Z(j) = rand_nort(m,q,0,inf);
        else
            Z(j) = rand_nort(m,q,-inf,0);
        end
        
        % Make change to B
        B = B + ((Z(j)-z_old)/Lambda(j))*S(:,j);
    end

    % Now draw new value of Beta
    T = mvnrnd(zeros(D,1),eye(D))';
    Beta = B + L*T;
    
    if i > BurnIn
        betaSaved(i-BurnIn,:) = Beta;
    end

    % Now draw new values for mixing variances
    for j = 1:N
        R = Z(j)-XX(j,:)*Beta;
        Lambda(j) = Sample_Lambda(R^2);
    end
end

% Stop timer
TimeTaken = toc;

betaPosterior = betaSaved;

CurTime = fix(clock);
cd('..')
save(['Results/Results_HolmesJointUpdate_' DataSet '_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '.mat'], 'betaPosterior', 'TimeTaken')

NumOfPlots = min(25,D);

for d =1:NumOfPlots
    figure(90)
    subplot(5,5,d)
    hist(betaSaved(:,d));
    drawnow
end

figure(91)
plot(betaSaved);
drawnow


end



function [Lambda] = Sample_Lambda(r2)

r = sqrt(r2);

OK = 0;

    while ~OK
        Y = randn;
        Y = Y*Y;
        Y = 1+(Y-sqrt(Y*(4*r+Y)))/(2*r);
        U = rand;

        if U <= 1/(1+Y)
            Lambda = r/Y;
        else
            Lambda = r*Y;
        end

        % Now sample Lambda ~ GIG(0.5,1,r^2)
        U = rand;

        if U > 4/3
            OK = RightmostInterval(U,Lambda);
        else
            OK = LeftmostInterval(U,Lambda);
        end
    end
    
end


function [OK] = RightmostInterval(U,Lambda)

    Z = 1;
    X = exp(-0.5*Lambda);
    j = 0;

    while 1
        j = j + 1;
        Z = Z-((j+1)^2)*X^(((j+1)^2) - 1);

        if Z > U
            OK = 1;
            return;
        end

        j = j + 1;
        Z = Z+((j+1)^2)*X^(((j+1)^2) - 1);

        if Z < U
            OK = 0;
            return;
        end

    end
    
end

function [OK] = LeftmostInterval(U,Lambda)
    
    H = 0.5*log(2) + 2.5*log(pi) - 2.5*log(Lambda) - (pi^2)/(2*Lambda) + 0.5*Lambda;
    lU = log(U);
    Z = 1;
    X = exp((-pi^2)/(2*Lambda));
    K = Lambda/(pi^2);
    j = 0;

    while 1
        j = j + 1;
        Z = Z - K*X^((j^2)-1);

        if H + log(Z) > lU
            OK = 1;
            return;
        end

        j = j + 1;
        Z = Z + ((j+1)^2)*X^(((j+1)^2) -1);

        if H + log(Z) < lU
            OK = 0;
            return;
        end
    end
end



