function [] = ODE_RMHMC( Y, TimePoints, Options )

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
CurrentG     = zeros(NumOfParameters);
CurrentGradL = zeros(1,NumOfParameters);

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


% Set up initial noise for likelihood
% Fix noise - CurrentNoise is the variance
CurrentNoise = ones(1,NumOfSpecies)*SDNoiseAdded^2;


% RMHMC Setup
NumOfLeapFrogSteps = 6;
StepSize           = 3/NumOfLeapFrogSteps;
NumOfFixPointSteps = 5;


% Set monitor rate for adapting step sizes
MonitorRate = 100;


% Set up converged flag
ContinueIterations = true;
Converged          = false;
warning('off')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Precalculate some values for speed                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Precalculate the log likelihoods
try
    %simdata = feval(SBModelNameSens2, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies) zeros(1,sum(1:NumOfParameters)*NumOfSpecies)], Parameters, options);
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
    
    % Save current gradient
    CurrentGradL = GradL;
    
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
    CurrentG     = G;
    CurrentCholG = chol(G); 
    CurrentInvG  = inv(G + eye(NumOfParameters)*1e-6);
    
    
    % Now calculate the partial derivatives of the metric tensor
    for k = 1:NumOfParameters
        CurrentGDeriv{k} = zeros(NumOfParameters);
        
        % Use sensitivities
        %
        % Standard derivatives of fisher expression for a gaussian
        for SpeciesNum_a = SpeciesObserved %1:NumOfSpecies
            for i = 1:NumOfParameters
                for j = i:NumOfParameters

                    CurrentGDeriv{k}(i,j) = CurrentGDeriv{k}(i,j) + (1/CurrentNoise(SpeciesNum_a))*( Sensitivities2{i,k}(:,SpeciesNum_a)'*Sensitivities{j}(:,SpeciesNum_a) )...
                                                                  + (1/CurrentNoise(SpeciesNum_a))*( Sensitivities{i}(:,SpeciesNum_a)'*Sensitivities2{j,k}(:,SpeciesNum_a) );
                    CurrentGDeriv{k}(j,i) = CurrentGDeriv{k}(i,j);
                end
            end
        end
        
        % Add prior to the FI: - 3rd derivative of log gamma in this case, Gam(3,1)
        CurrentGDeriv{k} = CurrentGDeriv{k} - diag(4*ones(1,NumOfParameters)./Parameters.^3);
        
        CurrentInvGdG{k}       = CurrentInvG*CurrentGDeriv{k};
        CurrentTraceInvGdG(k)  = trace(CurrentInvGdG{k});
        
    end
    
    
    
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
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Mutate parameter values %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    OriginalParas = Parameters;
    NewParas      = OriginalParas;
    
    AttemptedMutation = AttemptedMutation + 1;
    
    
    GradL = CurrentGradL;
    G     = CurrentG;
    CholG = CurrentCholG;
    InvG  = CurrentInvG;
    
    
    ProposedMomentum = (randn(1,NumOfParameters)*CholG)';
    OriginalMomentum = ProposedMomentum;
    
    
    InvGdG      = CurrentInvGdG;
    TraceInvGdG = CurrentTraceInvGdG;
    
    if (randn > 0.5) TimeStep = 1; else TimeStep = -1; end
    
    RandomSteps = ceil(rand*NumOfLeapFrogSteps);
    
    
    IntegrationErr = false;
    
    try
        
        % Perform leapfrog steps
        for StepNum = 1:RandomSteps

            %%%%%%%%%%%%%%%%%%%
            % Update momentum %
            %%%%%%%%%%%%%%%%%%%
            dJLdTheta = GradL;
            TraceTerm = 0.5*TraceInvGdG';

            % Multiple fixed point iteration
            PM = ProposedMomentum;
            for FixedIter = 1:NumOfFixPointSteps
                MomentumHist(FixedIter,:) = PM;

                InvGMomentum = InvG*PM;
                for d = 1:NumOfParameters
                    LastTerm(d)  = 0.5*(PM'*InvGdG{d}*InvGMomentum);
                end

                PM = ProposedMomentum + (TimeStep*StepSize/2)*(dJLdTheta' - TraceTerm + LastTerm');
            end
            ProposedMomentum = PM;




            %%%%%%%%%%%%%%%%%%%%%%%
            % Update w parameters %
            %%%%%%%%%%%%%%%%%%%%%%%
            %%% Multiple Fixed Point Iteration %%%
            OriginalInvGMomentum  = G\ProposedMomentum;

            Pw = NewParas;
            for FixedIter = 1:NumOfFixPointSteps
                wHist(FixedIter,:) = Pw;

                InvGMomentum = (G + eye(NumOfParameters)*1e-6)\ProposedMomentum;

                Pw = NewParas + (TimeStep*(StepSize/2))*OriginalInvGMomentum' + (TimeStep*(StepSize/2))*InvGMomentum';


                %%%%%%%%%%%%%%%
                % Calculate G %
                %%%%%%%%%%%%%%%
                if FixedIter == NumOfFixPointSteps
                    % Calculate 2nd order sensitivities if last iteration
                    %simdata          = feval(SBModelNameSens2, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies) zeros(1,sum(1:NumOfParameters)*NumOfSpecies)], Pw, options);
                    [TimeData,XData] = ode45(@FitzHughNagumoSens2, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies) zeros(1,sum(1:NumOfParameters)*NumOfSpecies)], ODEoptions, Pw);
                else
                    % Otherwise we just need G, so calculate 1st order sensitivities only
                    %simdata          = feval(SBModelNameSens1, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], Pw, options);
                    [TimeData,XData] = ode45(@FitzHughNagumoSens1, TimePoints, [InitialValues zeros(1,NumOfParameters*NumOfSpecies)], ODEoptions, Pw);
                end


                % Get sensitivities of all species with respect to parameter j
                for j = 1:NumOfParameters
                    Sensitivities{j} = XData(:,(NumOfSpecies*j)+1:(NumOfSpecies*(j+1)));
                end

                % Now calculate metric tensor
                G = zeros(NumOfParameters);

                for SpeciesNum = SpeciesObserved %1:NumOfSpecies
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
                G = G - diag(-2*ones(1,NumOfParameters)./Pw.^2);


            end
            NewParas = Pw;

            
            

            % Get second order sensitivities of all species with respect to parameters j and k
            for j = 1:NumOfParameters
                for k = j:NumOfParameters
                    CurrentStartIndex   = ( (NumOfSpecies*(NumOfParameters+1)) + (sum(1:NumOfParameters)-sum(1:NumOfParameters-(j-1)))*NumOfSpecies ) + (k-j)*NumOfSpecies + 1;
                    Sensitivities2{j,k} = XData(:, CurrentStartIndex:(CurrentStartIndex+(NumOfSpecies-1)));
                    Sensitivities2{k,j} = Sensitivities2{j,k};
                end
            end

            % Get species time series
            DataTemp = XData(:,1:NumOfSpecies)';

            % Get inverse metric tensor
            InvG = inv(G + eye(NumOfParameters)*1e-6);


            % Calculate gradient and G
            GradL = zeros(1, NumOfParameters);
            for ParaNum = 1:NumOfParameters
                for i = SpeciesObserved
                    GradL(ParaNum) = GradL(ParaNum) + sum( -((DataTemp(i,:)-Y(i,:)).*Sensitivities{ParaNum}(:,i)')./(ones(1,NumOfObs)*CurrentNoise(i)) );
                end
                GradL(ParaNum) = GradL(ParaNum) + ModelParameterLogPriorDerivative(ParaNum,NewParas(ParaNum));
            end

            % Now calculate the partial derivatives of the metric tensor
            for k = 1:NumOfParameters
                GDeriv{k} = zeros(NumOfParameters);

                % Use sensitivities
                %
                % Standard derivatives of fisher expression for a gaussian
                for SpeciesNum_a = SpeciesObserved %1:NumOfSpecies
                    for i = 1:NumOfParameters
                        for j = i:NumOfParameters

                            GDeriv{k}(i,j) = GDeriv{k}(i,j) + (1/CurrentNoise(SpeciesNum_a))*( Sensitivities2{i,k}(:,SpeciesNum_a)'*Sensitivities{j}(:,SpeciesNum_a) )...
                                                            + (1/CurrentNoise(SpeciesNum_a))*( Sensitivities{i}(:,SpeciesNum_a)'*Sensitivities2{j,k}(:,SpeciesNum_a) );
                            GDeriv{k}(j,i) = GDeriv{k}(i,j);
                        end
                    end
                end

                % Add prior to the FI: - 3rd derivative of log gamma in this case, Gam(3,1)
                GDeriv{k} = GDeriv{k} - diag(4*ones(1,NumOfParameters)./NewParas.^3);

                InvGdG{k}       = InvG*GDeriv{k};
                TraceInvGdG(k)  = trace(InvGdG{k});
            end



            %%%%%%%%%%%%%%%%%%%
            % Update momentum %
            %%%%%%%%%%%%%%%%%%%
            % Calculate last term in dH/dTheta
            dJLdTheta = GradL;
            TraceTerm = 0.5*TraceInvGdG';

            InvGMomentum = (InvG*ProposedMomentum);
            for d = 1:NumOfParameters
                LastTerm(d) = 0.5*((ProposedMomentum'*InvGdG{d}*InvGMomentum));
            end

            ProposedMomentum = ProposedMomentum + (TimeStep*StepSize/2)*(dJLdTheta' - TraceTerm + LastTerm');

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
        
        
        % Calculate proposed H value
        % Get species time series
        XEstimates = XData(:,1:NumOfSpecies)';
    
        % Calculate the current likelihoods of the current parameters
        for n=SpeciesObserved
            ProposedLL(n) = LogNormPDF(XEstimates(n,:), Y(n,:), CurrentNoise(n));
        end
    
        CholG = chol(G);
        ProposedLogDet = 0.5*( log(2) + NumOfParameters*log(pi) + 2*sum(log(diag(CholG))) );
        ProposedH = -(sum(ProposedLL) + sum(ProposedLogPrior)) + ProposedLogDet + (ProposedMomentum'*InvG*ProposedMomentum)/2;
        
        
        % Calculate current H value
        % Calculate the log prior for current hyperparameter value
        for a = 1:NumOfParameters
            CurrentLogPrior(a) = ModelParameterPrior(a, Parameters(a));
            if CurrentLogPrior(a) == 0
                CurrentLogPrior(a) = -1e300;
            else
                CurrentLogPrior(a) = log(CurrentLogPrior(a));
            end
        end
        
        CurrentLogDet = 0.5*( log(2) + NumOfParameters*log(pi) + 2*sum(log(diag(CurrentCholG))) );
        CurrentH      = -(sum(CurrentLL) + sum(CurrentLogPrior)) + CurrentLogDet + (OriginalMomentum'*CurrentInvG*OriginalMomentum)/2;
        
        % Accept according to ratio
        Ratio = -ProposedH + CurrentH;
        
        
        if Ratio > 0 || (Ratio > log(rand))
            % Accept proposal
            % Update variables
            Parameters                 = NewParas;
            CurrentLL(SpeciesObserved) = ProposedLL(SpeciesObserved);
            AcceptedMutation           = AcceptedMutation + 1;
            
            CurrentG           = G;
            CurrentGradL       = GradL;
            CurrentInvG        = InvG;
            CurrentCholG       = CholG;
            
            CurrentInvGdG      = InvGdG;
            CurrentTraceInvGdG = TraceInvGdG;
            
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
        
        if SaveMetricTensors
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
FileName = ['ODE_RMHMC_' EquationName '_' num2str(D) 'DPS_' num2str(floor(now)) '_' num2str(CurTime(4:6)) '_' num2str(RandTime)];
save(['./Results/' FileName], 'ParaHistory', 'MetricTensorHistory', 'LLHistory', 'BurnInTime', 'PosteriorTime', 'Y', 'TimePoints');



end
