function [] = ODE_MALA( Y, TimePoints, Options )

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

ODEoptions            = odeset('RelTol',1e-6,'AbsTol',1e-6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise non changeable stuff                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup initial parameters
Parameters = [0.2 0.2 3]; % Fitzhugh Nagumo

% Current Gradient
CurrentGradient = zeros(1,NumOfParameters);


% Setup initial values for solving the ODE
InitialValues   = Y(1,SpeciesObserved); % Initial observation points
InitialValues   = [-1 1]; % For fitzhugh nagumo


% Set up proposal counters
AcceptedMutation  = 0;
AttemptedMutation = 0;

% Set up parameter history variable
ParaHistory         = zeros(NumOfPosteriorSamples,NumOfParameters);
LLHistory           = zeros(NumOfPosteriorSamples,NumOfSpecies);


% Set up initial noise for likelihood
% Fix noise - CurrentNoise is the variance
CurrentNoise = ones(1,NumOfSpecies)*SDNoiseAdded^2;


StepSize = 0.0002;
Scaling  = NumOfParameters^(1/3);

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
    %simdata          = feval(SBModelName, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], Parameters, options);
    [TimeData,XData] = ode45(@FitzHughNagumoSens1, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], ODEoptions, Parameters);
    
    % Get sensitivities of all species with respect to parameter j
    for j = 1:NumOfParameters
        Sensitivities{j} = XData(:,(NumOfSpecies*j)+1:(NumOfSpecies*(j+1)));
    end
    
    % XEstimates species time series
    XEstimates = XData(:,1:NumOfSpecies)';
    
    % Calculate gradients for each of the parameters i.e. d(LL)/d(Parameter)
    GradL = zeros(1, NumOfParameters);
    for ParaNum = 1:NumOfParameters
        for i = SpeciesObserved %1:NumOfSpecies
            GradL(ParaNum) = GradL(ParaNum) + sum( -((XEstimates(i,:)-Y(i,:)).*Sensitivities{ParaNum}(:,i)')./((ones(1,NumOfObs)*CurrentNoise(i))) );
        end
        GradL(ParaNum) = GradL(ParaNum) + ModelParameterLogPriorDerivative(ParaNum,Parameters(ParaNum));
    end
    
    % Save current gradient
    CurrentGradient = GradL;
    
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
    
    
    
    GradL = CurrentGradient;
    
    
    Mean     = OriginalParas + (StepSize/(2*Scaling))*GradL;
    NewParas = Mean + randn(1,NumOfParameters)*sqrt((StepSize/Scaling));
    
    
    % Calculate proposed likelihood
    try
        %simdata = feval(SBModelName, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], NewParas, options);
        [TimeData,XData] = ode45(@FitzHughNagumoSens1, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], ODEoptions, NewParas);
        
        for j = 1:NumOfParameters
            % Get sensitivities of all species with respect to parameter j
            Sensitivities{j} = XData(:,(NumOfSpecies*j)+1:(NumOfSpecies*(j+1)));
        end
        
        XEstimates = XData(:,1:NumOfSpecies)';
        
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
    ProbNewGivenOld = LogNormPDF(Mean, NewParas,(StepSize/Scaling));
    
    
    % Calculate gradient and G
    GradL = zeros(1, NumOfParameters);
    for ParaNum = 1:NumOfParameters
        for i = SpeciesObserved
            GradL(ParaNum) = GradL(ParaNum) + sum( -((XEstimates(i,:)-Y(i,:)).*Sensitivities{ParaNum}(:,i)')./(ones(1,NumOfObs)*CurrentNoise(i)) );
        end
        GradL(ParaNum) = GradL(ParaNum) + ModelParameterLogPriorDerivative(ParaNum,NewParas(ParaNum));
    end
    
    
    Mean            = NewParas + (StepSize/(2*Scaling))*GradL;
    ProbOldGivenNew = LogNormPDF(OriginalParas, Mean, (StepSize/Scaling));
    
    
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
        
        
        if Ratio > 0 || (Ratio > log(rand))
            % Accept proposal
            % Update variables
            Parameters                 = NewParas;
            CurrentLL(SpeciesObserved) = ProposedLL(SpeciesObserved);
            AcceptedMutation           = AcceptedMutation + 1;
            
            CurrentGradient            = GradL;
            
            %disp('Accepted')
        else
            %disp('Rejected')
        end
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%
    % Save parameters %
    %%%%%%%%%%%%%%%%%%%
    if Converged
        ParaHistory(IterationNum-ConvergenceIterationNum, :) = Parameters;
        LLHistory(IterationNum-ConvergenceIterationNum, :)   = CurrentLL;
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
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Calculate convergence every 1000 steps %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Change converged tab if converged
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
FileName = ['ODE_MALA_' EquationName '_' num2str(D) 'DPS_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '_' num2str(RandTime)];
save(['./Results/' FileName], 'ParaHistory', 'LLHistory', 'BurnInTime', 'PosteriorTime', 'Y', 'TimePoints');



end
