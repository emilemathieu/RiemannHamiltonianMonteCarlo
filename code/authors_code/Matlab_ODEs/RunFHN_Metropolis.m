addpath(genpath('./'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fix random numbers for generating data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

randn('state', 1);
rand('twister', 1);


%%%%%%%%%%%%%%%
% Set options %
%%%%%%%%%%%%%%%

Options.Burnin                = 1000;
Options.NumOfPosteriorSamples = 5000;


% Set name for saving results
Options.EquationName          = 'FHN';

Options.NumOfParameters       = 3;
Options.ObservedSpecies       = [1 2];
Options.UnobservedSpecies     = [];

Options.SDNoiseAdded          = 0.5;

Options.SaveMetricTensors     = true;


%%%%%%%%%%%%%%%%%
% Generate Data %
%%%%%%%%%%%%%%%%%

% Simulate data
D             = 200;
StartTime     = 0;
EndTime       = 20;
TimePoints    = StartTime:EndTime/(D-1):EndTime;
InitialValues = [-1 1];
NoiseSD       = Options.SDNoiseAdded;
Parameters    = [0.2 0.2 3];
ODEoptions    = odeset('RelTol',1e-6,'AbsTol',1e-6);

[TimeData,NoisyData] = ode45(@FitzHughNagumo,TimePoints,InitialValues,ODEoptions,Parameters);

TimeData  = TimeData';
NoisyData = NoisyData';

[NumOfSpecies,D] = size(NoisyData);

% Add Noise
NoisyData = NoisyData + randn(NumOfSpecies, D).*NoiseSD;




%%%%%%%%%%%%%%%%%%%%%%%%%        
% Call sampling routine %
%%%%%%%%%%%%%%%%%%%%%%%%%

ODE_Metropolis(NoisyData, TimeData, Options);


