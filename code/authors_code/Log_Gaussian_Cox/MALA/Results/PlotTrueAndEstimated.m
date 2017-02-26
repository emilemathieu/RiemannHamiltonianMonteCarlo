function[] = PlotTrueAndEstimated(xEstimates, xVar)

% xEstimates is the mean of the samples of x - i.e. N^2 by 1 vector
% xVar is the variance of the the samples of x - i.e. N^2 by 1 vector



% Y is the data
% X is the latent field

% Load data
Data=load('TestData64.mat');
x = Data.X;
y = Data.Y;


figure(5)
Results = x;
subplot(2,3,1)
imagesc(reshape(Results,N,N));
subplot(2,3,2)
imagesc(reshape(exp(Results)./N^2,N,N));
subplot(2,3,3)
imagesc(reshape(y,N,N));

Results = xEstimates;
subplot(2,3,4)
imagesc(reshape(Results,N,N));
subplot(2,3,5)
imagesc(reshape(exp(Results)./N^2,N,N));
subplot(2,3,6)
imagesc(reshape(xVar,N,N));




end