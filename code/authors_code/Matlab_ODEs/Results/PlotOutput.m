function [ ] = PlotOutput( PosteriorSamples, TimePoints )

[NumOfSamples, NumOfParameters] = size(PosteriorSamples);

InitialValues = [-1 1];
ODEoptions    = odeset('RelTol',1e-6,'AbsTol',1e-6);

figure(2)
hold on

for SampleNum = 1:NumOfSamples
    
    [t,XEstimates] = ode45(@FitzHughNagumo,TimePoints,InitialValues,ODEoptions,PosteriorSamples(SampleNum, :));
    
    plot(t, XEstimates)
    drawnow
    
end


end

