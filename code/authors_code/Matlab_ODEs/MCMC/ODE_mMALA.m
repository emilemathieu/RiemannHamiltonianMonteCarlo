function [] = ODE_mMALA( Y, TimePoints, Options )

% Start the timer
tic

rand('twister', sum(100*clock))
randn('state', sum(100*clock))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise user options if not already specified                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get N (number of chemical species) and D (number of time points)
[N, D] = size(Y);
NumOfSpecies = N;
NumOfObs     = D;

% Get specified options
NumOfParameters       = Options.NumOfParameters;

SpeciesObserved       = Options.ObservedSpecies;
SpeciesUnobserved     = Options.UnobservedSpecies;
SDNoiseAdded          = Options.SDNoiseAdded;

N = length(SpeciesObserved) + length(SpeciesUnobserved);

Burnin                = Options.Burnin;
NumOfPosteriorSamples = Options.NumOfPosteriorSamples;

EquationName          = Options.EquationName;

SaveMetricTensors     = Options.SaveMetricTensors;

ODEoptions            = odeset('RelTol',1e-6,'AbsTol',1e-6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise non changeable stuff                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup initial parameters
Parameters = [0.2 0.2 3]; % Fitzhugh Nagumo

% Current Gradient
CurrentG = zeros(NumOfParameters);


% Setup initial values for solving the ODE
%InitialValues   = Y(1,SpeciesObserved); % Initial observation points
InitialValues   = [-1 1]; % For fitzhugh nagumo


% Set up proposal counters
AcceptedMutation  = 0;
AttemptedMutation = 0;

% Set up parameter history variable
ParaHistory         = zeros(NumOfPosteriorSamples,NumOfParameters);
LLHistory           = zeros(NumOfPosteriorSamples,NumOfSpecies);

MetricTensorHistory = cell(1,NumOfPosteriorSamples);


% Set up noise for likelihood function
% Fix noise - CurrentNoise is the variance
CurrentNoise = ones(1,NumOfSpecies)*SDNoiseAdded^2;


StepSize = 1;


% Set monitor rate for adapting step sizes
MonitorRate = 100;

% Set up converged flag
ContinueIterations = true;
Converged          = false;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Precalculate some values for speed                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Precalculate the log likelihood
try
    
    %simdata          = feval(SBModelName, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies) zeros(1,sum(1:NumOfParameters)*NumOfSpecies)], Parameters, options);
    [TimeData,XData] = ode45(@FitzHughNagumoSens2, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies) zeros(1,sum(1:NumOfParameters)*NumOfSpecies)], ODEoptions, Parameters);
    
    % Get sensitivities of all species with respect to parameter j
    for j = 1:NumOfParameters
        Sensitivities{j} = XData(:,(NumOfSpecies*j)+1:(NumOfSpecies*(j+1)));
        
        % Get second order sensitivities of all species with respect to parameters j and k
        for k = j:NumOfParameters
            CurrentStartIndex   = ( (NumOfSpecies*(NumOfParameters+1)) + (sum(1:NumOfParameters)-sum(1:NumOfParameters-(j-1)))*NumOfSpecies ) + (k-j)*NumOfSpecies + 1;
            Sensitivities2{j,k} = XData(:, CurrentStartIndex:(CurrentStartIndex+(NumOfSpecies-1)));
            Sensitivities2{k,j} = Sensitivities2{j,k};
        end
        
    end
    
    
    % Get species time series
    DataTemp         = XData(:,1:NumOfSpecies)';
    XEstimates       = DataTemp;
    
    
    % Calculate gradients for each of the parameters i.e. d(LL)/d(Parameter)
    GradL = zeros(1, NumOfParameters);
    for ParaNum = 1:NumOfParameters
        for i = SpeciesObserved
            GradL(ParaNum) = GradL(ParaNum) + sum( -((DataTemp(i,:)-Y(i,:)).*Sensitivities{ParaNum}(:,i)')./((ones(1,NumOfObs)*CurrentNoise(i))) );
        end
        GradL(ParaNum) = GradL(ParaNum) + ModelParameterLogPriorDerivative(ParaNum,Parameters(ParaNum));
    end
    
    
    % Now calculate metric tensor
    G = zeros(NumOfParameters);
    
    for SpeciesNum = SpeciesObserved
        for i = 1:NumOfParameters
            for j = i:NumOfParameters
                G(i,j) = G(i,j) + (1/CurrentNoise(SpeciesNum))*(Sensitivities{i}(:,SpeciesNum)'*Sensitivities{j}(:,SpeciesNum));
            end
        end
    end
    
    for i = 1:NumOfParameters
        for j = i:NumOfParameters
            G(j,i) = G(i,j);
        end
    end
    
    % Add prior to the FI: - 2nd derivative of log gamma in this case, Gam(3,1)
    G = G - diag(-2*ones(1,NumOfParameters)./Parameters.^2);
    
    % Save current metric tensor
    CurrentG    = G;
    CurrentInvG = inv(G + eye(NumOfParameters)*1e-6);
    
    
    CurrentFirstTerm = (CurrentInvG*GradL')';
    
    % Now calculate the partial derivatives of the metric tensor
    for k = 1:NumOfParameters
        GDeriv{k} = zeros(NumOfParameters);
        
        % Use sensitivities
        % Standard derivatives of Fisher Information for a gaussian
        for SpeciesNum_a = SpeciesObserved
            for i = 1:NumOfParameters
                for j = 1:NumOfParameters

                    GDeriv{k}(i,j) = GDeriv{k}(i,j) + (1/CurrentNoise(SpeciesNum_a))*( Sensitivities2{i,k}(:,SpeciesNum_a)'*Sensitivities{j}(:,SpeciesNum_a) )...
                                                    + (1/CurrentNoise(SpeciesNum_a))*( Sensitivities{i}(:,SpeciesNum_a)'*Sensitivities2{j,k}(:,SpeciesNum_a) );

                end
            end
        end
        
        % Add prior to the FI: - 3rd derivative of log gamma in this case, Gam(3,1)
        GDeriv{k} = GDeriv{k} - diag(4*ones(1,NumOfParameters)./Parameters.^3);
        
        
        InvGdG{k} = CurrentInvG*GDeriv{k};
        TraceInvGdG(k) = trace(InvGdG{k});
        
        CurrentSecondTerm(:,k) = InvGdG{k}*CurrentInvG(:,k);
        
    end
    
    CurrentThirdTerm = CurrentInvG*TraceInvGdG';
    
    
    % Calculate the current likelihoods of the current parameters
    for n=SpeciesObserved
        CurrentLL(n) = LogNormPDF(XEstimates(n,:), Y(n,:), CurrentNoise(n));
    end
    
catch
    
    for n=1:SpeciesObserved
        CurrentLL(n) = -1e300;
    end
    
end



disp('Initialisation Completed..');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Manifold MALA Algorithm to sample the parameters                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Initialise iteration number
IterationNum = 0;

% Main loop
while ContinueIterations
    
    % Increment iteration number
    IterationNum = IterationNum + 1;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Mutate parameter values %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    OriginalParas = Parameters;
    
    AttemptedMutation = AttemptedMutation + 1;
    
    
    G     = CurrentG;
    InvG  = CurrentInvG;
    
    
    Mean  = OriginalParas + (StepSize/2)*CurrentFirstTerm...
                          - StepSize*sum(CurrentSecondTerm,2)'...
                          + (StepSize/2)*CurrentThirdTerm';
    
    NewParas = Mean + ( randn(1,NumOfParameters)*chol(StepSize*InvG) );
    
    
    % Calculate proposed likelihood
    try
        %simdata = feval(SBModelName, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies) zeros(1,sum(1:NumOfParameters)*NumOfSpecies)], NewParas, options);
        [TimeData,XData] = ode45(@FitzHughNagumoSens2, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies) zeros(1,sum(1:NumOfParameters)*NumOfSpecies)], ODEoptions, NewParas);
        
        for j = 1:NumOfParameters
            
            % Get sensitivities of all species with respect to parameter j
            Sensitivities{j} = XData(:,(NumOfSpecies*j)+1:(NumOfSpecies*(j+1)));
        
            % Get second order sensitivities of all species with respect to parameters j and k
            for k = j:NumOfParameters
                CurrentStartIndex   = ( (NumOfSpecies*(NumOfParameters+1)) + (sum(1:NumOfParameters)-sum(1:NumOfParameters-(j-1)))*NumOfSpecies ) + (k-j)*NumOfSpecies + 1;
                Sensitivities2{j,k} = XData(:, CurrentStartIndex:(CurrentStartIndex+(NumOfSpecies-1)));
                Sensitivities2{k,j} = Sensitivities2{j,k};
            end
            
        end
        
        
        DataTemp         = XData(:,1:NumOfSpecies)';
        XEstimates       = DataTemp;
        
        % Calculate the current likelihoods of the current X estimates
        for n=SpeciesObserved
            ProposedLL(n) = LogNormPDF(XEstimates(n,:), Y(n,:), CurrentNoise(n));
        end
    catch
        for n=SpeciesObserved
            ProposedLL(n) = -1e300;
        end
    end
    
    % Calculate the log prior for current parameter value
    for a = 1:NumOfParameters
        ProposedLogPrior(a) = ModelParameterPrior(a, NewParas(a));
    end
    
    % Calculate new given old
    ProbNewGivenOld = -sum(log(diag(chol(StepSize*InvG)))) - 0.5*(Mean-NewParas)*(G/StepSize)*(Mean-NewParas)';
    
    
    
    % Calculate gradient and G
    GradL = zeros(1, NumOfParameters);
    for ParaNum = 1:NumOfParameters
        for i = SpeciesObserved
            GradL(ParaNum) = GradL(ParaNum) + sum( -((DataTemp(i,:)-Y(i,:)).*Sensitivities{ParaNum}(:,i)')./(ones(1,NumOfObs)*CurrentNoise(i)) );
        end
        GradL(ParaNum) = GradL(ParaNum) + ModelParameterLogPriorDerivative(ParaNum,NewParas(ParaNum));
    end
    
    G = zeros(NumOfParameters);
    
    for SpeciesNum = SpeciesObserved
        for i = 1:NumOfParameters
            for j = i:NumOfParameters
                G(i,j) = G(i,j) + (1/CurrentNoise(SpeciesNum))*(Sensitivities{i}(:,SpeciesNum)'*Sensitivities{j}(:,SpeciesNum));
            end
        end
    end
    
    for i = 1:NumOfParameters
        for j = i:NumOfParameters
            G(j,i) = G(i,j);
        end
    end
    
    G    = G - diag(-2*ones(1,NumOfParameters)./NewParas.^2);
    InvG = inv(G + eye(NumOfParameters)*1e-6);
    
    
    FirstTerm = (InvG*GradL')';
    
    % Now calculate the partial derivatives of the metric tensor
    for k = 1:NumOfParameters
        
        GDeriv{k} = zeros(NumOfParameters);
        
        % Use sensitivities
        % Standard derivatives of fisher expression for a gaussian
        for SpeciesNum_a = SpeciesObserved %1:NumOfSpecies
            for i = 1:NumOfParameters
                for j = 1:NumOfParameters

                    GDeriv{k}(i,j) = GDeriv{k}(i,j) + (1/CurrentNoise(SpeciesNum_a))*( Sensitivities2{i,k}(:,SpeciesNum_a)'*Sensitivities{j}(:,SpeciesNum_a) )...
                                                    + (1/CurrentNoise(SpeciesNum_a))*( Sensitivities{i}(:,SpeciesNum_a)'*Sensitivities2{j,k}(:,SpeciesNum_a) );

                end
            end
        end

        % Add prior to the FI: - 3rd derivative of log gamma in this case, Gam(3,1)
        GDeriv{k} = GDeriv{k} - diag(4*ones(1,NumOfParameters)./NewParas.^3);
        
        
        InvGdG{k}      = InvG*GDeriv{k};
        TraceInvGdG(k) = trace(InvGdG{k});
        
        SecondTerm(:,k) = InvGdG{k}*InvG(:,k);
        
    end
    
    ThirdTerm = InvG*TraceInvGdG';
    
    Mean      = NewParas + (StepSize/2)*FirstTerm...
                         - StepSize*sum(SecondTerm,2)'...
                         + (StepSize/2)*ThirdTerm';
                         
    ProbOldGivenNew = -sum(log(diag(chol(StepSize*InvG)))) - 0.5*(Mean-OriginalParas)*(G/StepSize)*(Mean-OriginalParas)';
    
    
    if min(ProposedLogPrior) > 0
        
        ProposedLogPrior = log(ProposedLogPrior);
        
        % Calculate the log prior for current hyperparameter value
        for a = 1:NumOfParameters
            CurrentLogPrior(a) = ModelParameterPrior(a, Parameters(a));
            if CurrentLogPrior(a) == 0
                CurrentLogPrior(a) = -1e300;
            else
                CurrentLogPrior(a) = log(CurrentLogPrior(a));
            end
        end
        
        % Accept according to ratio
        Ratio = sum(ProposedLL(SpeciesObserved)) + sum(ProposedLogPrior) + ProbOldGivenNew - sum(CurrentLL(SpeciesObserved)) - sum(CurrentLogPrior) - ProbNewGivenOld;
        
        
        if Ratio > 0 || log(rand) < min(0, Ratio)
            % Accept proposal
            % Update variables
            Parameters                 = NewParas;
            CurrentLL(SpeciesObserved) = ProposedLL(SpeciesObserved);
            AcceptedMutation           = AcceptedMutation + 1;
            
            CurrentG                   = G;
            CurrentInvG                = InvG;
            
            CurrentFirstTerm           = FirstTerm;
            CurrentSecondTerm          = SecondTerm;
            CurrentThirdTerm           = ThirdTerm;
            
            %disp('Accepted')
        else
            %disp('Rejected')
        end
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%
    % Save parameters %
    %%%%%%%%%%%%%%%%%%%
    if Converged

        ParaHistory(IterationNum-ConvergenceIterationNum, :)         = Parameters;
        LLHistory(IterationNum-ConvergenceIterationNum, :)           = CurrentLL;
        
        if SaveMetricTensors && Converged
            MetricTensorHistory{IterationNum-ConvergenceIterationNum} = CurrentG;
        end

    end
    
    % If not yet converged...
    if Converged == false
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Adjust proposal widths %
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Adjust parameter proposal widths
        if mod(IterationNum, MonitorRate) == 0
            
            disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            disp(['Iteration ' num2str(IterationNum)]);
            disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            disp(' ')
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Adjust proposal width for parameter value inference %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if AttemptedMutation > 0
                disp([num2str(100*AcceptedMutation/AttemptedMutation) '% mutation acceptance']);
            end
            
            disp(' ')
            
            % Reset counters
            AttemptedMutation = 0;
            AcceptedMutation  = 0;
            
            
        end
        
        
        
        % Change converged tab if burn-in complete
            
        if IterationNum >= Burnin
            Converged               = true;
            ConvergenceIterationNum = IterationNum;
            disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            disp(['Converged at iteration number ' num2str(ConvergenceIterationNum)]);
            disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            
            BurnInTime = toc;
            tic;
            
        end
        
        
    else % Converged so decide how long to sample from posteriors
        
        if IterationNum == ConvergenceIterationNum + NumOfPosteriorSamples
            % 5000 posterior samples have been collected so stop
            ContinueIterations = false;
        end
        
    end
    
    
end


PosteriorTime = toc;

CurTime = fix(clock);
RandTime = ceil(rand*10000);

% Save posterior
FileName = ['ODE_mMALA_' EquationName '_' num2str(D) 'DPS_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '_' num2str(RandTime)];
save(['./Results/' FileName], 'ParaHistory', 'MetricTensorHistory', 'LLHistory', 'BurnInTime', 'PosteriorTime', 'Y', 'TimePoints');



end
