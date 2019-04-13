clc
clear all
close all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
[f o]=size(T);
x_input=T{:,4:21};
[n m]=size(x_input);
Corr_x = corr(x_input);
x_cov=cov(x_input);
K = 0;
Alpha=0.01;
lamda=0.001;
% Normalisation 
for w=1:m
    if max(abs(x_input(:,w)))~=0;
        x_input(:,w)=(x_input(:,w)-mean((x_input(:,w))))./std(x_input(:,w));
        
    end
end
%{
The diagonal of the matrix contains the covariance between each variable and itself.
The other values in the matrix represent the covariance between
the two variables; in this case, the remaining two values are the same given
that we are calculating the covariance for only two variables.
%}
[U S V] = svd(x_cov); %Returns the eigenvectors U, the eigenvalues diag. in S;
%use S to find K

alpha=0.5;
while (alpha>=0.001)
    K=K+1;
    lamdas(K,:)=sum(max(S(:,1:K)));
    lamdass=sum(max(S));
    alpha=1-lamdas./lamdass;    
end
R=U(:, 1:K)'*(x_input)';
app_data=U(:,1:K)*R;
error=(1/m)*(sum(app_data-x_input'));
%%%LINEAR REGRESSION%%%%%%
h=1;
Theta=zeros(m,1);
k=1;
Y=T{:,3}/mean(T{:,3});
E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2); %cost function
while h==1
    Alpha=Alpha*1;
    Theta=Theta-(Alpha/m)*app_data*(app_data'*Theta-Y);
    k=k+1;
    E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2);
    
    %Regularization
    Reg(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2)+(lamda/(2*m))*sum(Theta.^2);
    %
    if E(k-1)-E(k)<0;
        break
    end
    q=(E(k-1)-E(k))./E(k-1);
    if q <.000001;
        h=0;
    end
end
%}

%%%%%%K_MEANS_CLUSTER%%%%%%%%%

idx = randperm(size(x_input, 1));
centroids = x_input(idx(1:K), :);
centroids = zeros(K, n);
for k = 1 : K,
	id = find(idx == k);
	centroids(k,:) = sum(x_input(id , :)) / size(id , 1);
end




