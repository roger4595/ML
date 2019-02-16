close all
clc
records = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA','MissingValue',0,'readsize',1800);
data = read(records);
price= table2array(data(1:1080,3)); %price
price=(price-mean(price))./std(price); %Normalizing output

% TRYING 4 DIFFERENT HYPOTHESIS

features=table2array((data(1:1080,4:9))) % Best one
%features=table2array((data(:,12:16)));
%features=table2array((data(:,6:15)));
%features=table2array((data(:,10:16)));

m = length(features);

% Normalization

for i=1:6
features(:,i)=(features(:,i)- mean(features(:,i)))./(std(features(:,i))); %normalizing inputs
end

% TRYING 4 DIFFERENT HYPOTHESIS

%features=[ones(m,1) features];
features=[ones(m,1) features features.^2]; % Best one
%features=[ones(m,1) features features.^2 features.^3];

thetas=randn(size(features,2),1); 
alpha=0.001; 
iterations=1000; 
mse=[]; 
i=[]; 

 for j=1:iterations
   hypothesis= features*thetas;
   cost=(1/(2*m))*sum((hypothesis-price).^2);
   mse=[mse;cost];
   i=[i;j];  
   thetas_new= Grad(features,m,hypothesis,price,thetas,alpha);
   thetas=thetas_new;
 
 end
 
% Normal Equation
 
thetaNormEqn = NormalEquation(features,price);
 
figure (1)
plot(i,mse)
title('House Price')
xlabel('Iterations')
ylabel('Cost Function')


