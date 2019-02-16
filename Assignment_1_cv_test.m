close all
clc
records = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA','MissingValue',0,'readsize',1800);
data = read(records);

price_training= table2array(data(1:1200,3)); %price
price_training=(price_training-mean(price_training))./std(price_training); %Normalizing output

price_cv= table2array(data(1201:1500,3)); %price
price_cv=(price_cv-mean(price_cv))./std(price_cv); %Normalizing output


features_training=table2array((data(1:1200,6:10)))

features_cv=table2array((data(1201:1500,6:10)))

m_training = length(features_training);

m_cv = length(features_cv);

% Normalization

for i=1:5
features_training(:,i)=(features_training(:,i)- mean(features_training(:,i)))./(std(features_training(:,i)));    
features_cv(:,i)=(features_cv(:,i)- mean(features_cv(:,i)))./(std(features_cv(:,i))); %normalizing inputs
end

features_training=[ones(m_training,1) features_training features_training.^2]; 
features_cv=[ones(m_cv,1) features_cv features_cv.^2 ]; 

thetas_training=randn(size(features_training,2),1); 
thetas_cv=randn(size(features_cv,2),1); 
alpha=0.001; 
iterations=1000; 
mse_cv=[]; 
mse_training=[];
i=[]; 

 for j=1:iterations
     
   hypothesis_training= features_training*thetas_training;  
   hypothesis_cv= features_cv*thetas_cv;
   
   cost_training=(1/(2*m_training))*sum((hypothesis_training-price_training).^2);
   cost_cv=(1/(2*m_cv))*sum((hypothesis_cv-price_cv).^2);
   
   mse_training=[mse_training;cost_training];
   mse_cv=[mse_cv;cost_cv];
   
   i=[i;j];  
   
   thetas_new_training= Grad(features_training,m_training,hypothesis_training,price_training,thetas_training,alpha);
   thetas_training=thetas_new_training;
   
   thetas_new_cv= Grad(features_cv,m_cv,hypothesis_cv,price_cv,thetas_cv,alpha);
   thetas_cv=thetas_new_cv;
 
 end
 
 
figure (1)
plot(i,mse_cv,'b')
hold on
plot(i,mse_training,'g')
legend('Cross Validation','Training Set')
title('House Price')
xlabel('Iterations')
ylabel('Cost Function')


