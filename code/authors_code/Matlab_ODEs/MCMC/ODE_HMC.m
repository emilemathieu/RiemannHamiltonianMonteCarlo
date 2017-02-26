function [] = ODE_HMC( Y, TimePoints, Options )

% Start the timer
tic

% Random numbers...
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
CurrentGradL = zeros(NumOfParameters);


% Setup initial values for solving the ODE
%InitialValues   = Y(1,SpeciesObserved); % Initial observation points
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


% HMC Setup
NumOfLeapFrogSteps = 150;
StepSize           = 1/NumOfLeapFrogSteps;


% Set monitor rate for adapting step sizes
MonitorRate = 100;


% Set up converged flag
ContinueIterations = true;
Converged          = false;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Precalculate some values for speed                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Precalculate the log likelihoods
try
    
    %simdata = feval(SBModelName, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], Parameters, options);
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
    CurrentGradL = GradL;
    
    
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
% Population MCMC Algorithm to sample the parameters based on likelihood  %
% of X given current parameters                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Initialise iteration number
IterationNum = 0;

% Main loop
while ContinueIterations
    
    % Increment iteration number
    IterationNum = IterationNum + 1;
    
    disp(IterationNum)
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Mutate parameter values %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    OriginalParas = Parameters;
    NewParas      = OriginalParas;
    
    AttemptedMutation = AttemptedMutation + 1;
    
    
    
    GradL = CurrentGradL;
    
    ProposedMomentum = randn(1,NumOfParameters);
    OriginalMomentum = ProposedMomentum;
    
    
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
    
    RandomSteps = ceil(rand*NumOfLeapFrogSteps);
    
    
    IntegrationErr = false;
    
    try
        % Perform leapfrog steps
        for StepNum = 1:RandomSteps

            %%%%%%%%%%%%%%%%%%%
            % Update momentum %
            %%%%%%%%%%%%%%%%%%%
            ProposedMomentum = ProposedMomentum + (TimeStep*StepSize/2)*GradL;


            %%%%%%%%%%%%%%%%%%%%%%%
            % Update w parameters %
            %%%%%%%%%%%%%%%%%%%%%%%
            % Unit mass matrix - therefore doesn't appear in equation below
            NewParas = NewParas + (TimeStep*StepSize)*ProposedMomentum;


            % Recalculate gradient
            %simdata          = feval(SBModelName, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], NewParas, options);
            [TimeData,XData] = ode45(@FitzHughNagumoSens1, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], ODEoptions, NewParas);
            
            % Get sensitivities of all species with respect to parameter j
            for j = 1:NumOfParameters
                Sensitivities{j} = XData(:,(NumOfSpecies*j)+1:(NumOfSpecies*(j+1)));
            end

            XEstimates = XData(:,1:NumOfSpecies)';

            % Calculate gradient
            GradL = zeros(1, NumOfParameters);
            for ParaNum = 1:NumOfParameters
                for i = SpeciesObserved
                    GradL(ParaNum) = GradL(ParaNum) + sum( -((XEstimates(i,:)-Y(i,:)).*Sensitivities{ParaNum}(:,i)')./(ones(1,NumOfObs)*CurrentNoise(i)) );
                end
                GradL(ParaNum) = GradL(ParaNum) + ModelParameterLogPriorDerivative(ParaNum,NewParas(ParaNum));
            end


            %%%%%%%%%%%%%%%%%%%
            % Update momentum %
            %%%%%%%%%%%%%%%%%%%
            ProposedMomentum = ProposedMomentum + (TimeStep*StepSize/2)*GradL;

        end
    catch
        IntegrationErr = true;
    end
        
        
    % Calculate the log prior for current hyperparameter value
    for a = 1:NumOfParameters
        ProposedLogPrior(a) = ModelParameterPrior(a, NewParas(a));
    end
        
        
    
    if min(ProposedLogPrior) > 0 && IntegrationErr == false
        
        ProposedLogPrior = log(ProposedLogPrior);
        
        
        % Get species time series
        %XEstimates = simdata.statevalues(:,1:NumOfSpecies)';
    
        % Calculate the current likelihoods of the current parameters
        for n=SpeciesObserved
            ProposedLL(n) = LogNormPDF(XEstimates(n,:), Y(n,:), CurrentNoise(n));
        end
        
        
        % Again unit mass matrix - therefore does not appear in equation
        ProposedH = -(sum(ProposedLL) + sum(ProposedLogPrior)) + (ProposedMomentum*ProposedMomentum')/2;

        
        % Calculate the log prior for current hyperparameter value
        for a = 1:NumOfParameters
            CurrentLogPrior(a) = ModelParameterPrior(a, Parameters(a));
            if CurrentLogPrior(a) == 0
                CurrentLogPrior(a) = -1e300;
            else
                CurrentLogPrior(a) = log(CurrentLogPrior(a));
            end
        end
        
        % Again unit mass matrix - therefore does not appear in equation
        CurrentH  = -(sum(CurrentLL) + sum(CurrentLogPrior)) + (OriginalMomentum*OriginalMomentum')/2;
    
        
        % Accept according to ratio
        Ratio = -ProposedH + CurrentH;
        
        
        if Ratio > 0 || (Ratio > log(rand))
            % Accept proposal
            % Update variables
            Parameters                 = NewParas;
            CurrentLL(SpeciesObserved) = ProposedLL(SpeciesObserved);
            AcceptedMutation           = AcceptedMutation + 1;
            
            CurrentGradL               = GradL;
            
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
        

        
        % Change converged tab if converged
        if IterationNum >= Burnin && Converged == false
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
FileName = ['ODE_HMC_' EquationName '_' num2str(D) 'DPS_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '_' num2str(RandTime)];
save(['./Results/' FileName], 'ParaHistory', 'LLHistory', 'BurnInTime', 'PosteriorTime', 'Y', 'TimePoints');



end
