function [] = ODE_Metropolis( Y, TimePoints, Options )

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

ODEoptions         = odeset('RelTol',1e-6,'AbsTol',1e-6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise non changeable stuff                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup initial parameters
Parameters = [0.2 0.2 3]; % Fitzhugh Nagumo

% Setup initial values for solving the ODE
%InitialValues   = Y(1,SpeciesObserved); % Initial observation points
InitialValues   = [-1 1]; % For fitzhugh nagumo


% Set up proposal counters
AcceptedMutation  = zeros(1,NumOfParameters);
AttemptedMutation = zeros(1,NumOfParameters);

% Set parameter step sizes for parameters in each chain in each population
ParameterWidth = ones(1,NumOfParameters)*0.1;

% Set up parameter history variable
ParaHistory         = zeros(NumOfPosteriorSamples, NumOfParameters);
LLHistory           = zeros(NumOfPosteriorSamples, NumOfSpecies);


% Set up initial noise for likelihood
% Fix noise - CurrentNoise is the variance
CurrentNoise = ones(1,NumOfSpecies)*SDNoiseAdded^2;

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
    
    %simdata = feval(SBModelName, TimePoints, InitialValues, Parameters, options);
    [TimeData,XEstimates] = ode45(@FitzHughNagumo,TimePoints,InitialValues,ODEoptions,Parameters);
    TimeData   = TimeData'; 
    XEstimates = XEstimates';
    % Get species time series
    %XEstimates = simdata.statevalues(:,1:NumOfSpecies)';
    
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
    % For each parameter, sample new value and accept with metropolis ratio
    for a = 1:NumOfParameters
        
        NewParas    = Parameters;
        NewParas(a) = Parameters(a) + randn*ParameterWidth(a);
        
        AttemptedMutation(a)  = AttemptedMutation(a) + 1;
        
        % Calculate the log prior for proposed model parameter
        ProposedLogPrior = ModelParameterPrior(a, NewParas(a));
        
        if ProposedLogPrior > 0
            
            ProposedLogPrior = log(ProposedLogPrior);
            
            try
                %simdata    = feval(SBModelName, TimePoints, InitialValues, NewParas, options);
                %t          = simdata.time;
                %XEstimates = simdata.statevalues';
                
                [t,XEstimates] = ode45(@FitzHughNagumo,TimePoints,InitialValues,ODEoptions,NewParas);
                t          = t'; 
                XEstimates = XEstimates';
                
                % Calculate the current likelihoods of the current X estimates
                for n=SpeciesObserved
                    ProposedLL(n) = LogNormPDF(XEstimates(n,:), Y(n,:), CurrentNoise(n));
                end
            catch
                for n=SpeciesObserved
                    ProposedLL(n) = -1e300;
                end
            end
            
            % Calculate the log prior for current hyperparameter value
            CurrentLogPrior = ModelParameterPrior(a, Parameters(a));
            if CurrentLogPrior == 0
                CurrentLogPrior = -1e300;
            else
                CurrentLogPrior = log(CurrentLogPrior);
            end
            
            Ratio = sum(ProposedLL) + ProposedLogPrior - sum(CurrentLL) - CurrentLogPrior;
            
            if Ratio > 0 || (Ratio > log(rand))
                % Accept proposal
                % Update variables
                Parameters                 = NewParas;
                CurrentLL(SpeciesObserved) = ProposedLL(SpeciesObserved);
                AcceptedMutation(a)        = AcceptedMutation(a) + 1;
            end
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
            for a = 1:NumOfParameters
                
                if AttemptedMutation(a) > 0
                    if AcceptedMutation(a)/AttemptedMutation(a) < 0.2
                        ParameterWidth(a) = ParameterWidth(a) * 0.9;
                    elseif AcceptedMutation(a)/AttemptedMutation(a) > 0.5
                        ParameterWidth(a) = ParameterWidth(a) * 1.1;
                    end
                    
                    disp([num2str(100*AcceptedMutation(a)/AttemptedMutation(a)) '% mutation acceptance for parameter ' num2str(a)]);
                end
                
            end
            
            disp(' ')
            disp('Parameter proposal widths:')
            disp(ParameterWidth)
            
            % Reset counters
            AttemptedMutation = zeros(1, NumOfParameters);
            AcceptedMutation  = zeros(1, NumOfParameters);
            
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
FileName = ['ODE_Metropolis_' EquationName '_' num2str(D) 'DPS_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '_' num2str(RandTime)];
save(['./Results/' FileName], 'ParaHistory', 'LLHistory', 'ParameterWidth', 'BurnInTime', 'PosteriorTime', 'Y', 'TimePoints');



end
