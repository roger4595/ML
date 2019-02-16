clc
clear all
close all
ds =datastore('heart_DD.csv','TreatAsMissing','NA','MissingValue',0,'readsize',250);
T = read(ds);
size(T);

Alpha=0.01;
%;
m=length(T{:,1});
U0=T{:,2};
U=T{:,1:13};


X=[ones(m,1) U U.^2]; 

n=length(X(1,:));
for w=2:n
    if max(abs(X(:,w)))~=0;
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
   
    end
end
lamda=0.25;


y=T{:,14}/mean(T{:,14});
thetas=zeros(n,1);
k=1;

for j=1:length(thetas)
r=exp(-X*thetas);
h=1./(1+(r));
g(j)=(1/m)*sum((h-y)'*X(:,j));

thetas=thetas-(Alpha/m)*X'*(log(h)-y);
E(k)=-(1/m)*sum((y.*log(h))+(1-y).*log(1-h))+(lamda/(2*m))*sum(thetas.^2);

end


mtrain=150;
mtest=(250-mtrain)/2;
mCV=(250-mtrain)/2;



U_trainSET=T{1:mtrain,1:13};
UCV=T{mtrain+1:mtrain+mCV,1:13};
U_testSET=T{mtrain+mCV+1:end,1:13};


lamda2=600;
x1=[ones(mtrain,1) U_trainSET U_trainSET.^2]; 
x2=[ones(mtest,1) U_testSET U_testSET.^2]; 
x3=[ones(mCV,1) UCV UCV.^2]; 

n1=length(x1(1,:));
n2=length(x2(1,:));
n3=length(x3(1,:));

thetas1=zeros(n1,1);


for w1=2:n1
    if max(abs(x1(:,w1)))~=0;
    x1(:,w1)=(x1(:,w1)-mean((x1(:,w1))))./std(x1(:,w1));
   
    end
end
for w2=2:n2
    if max(abs(x2(:,w2)))~=0;
    x2(:,w2)=(x2(:,w2)-mean((x2(:,w2))))./std(x2(:,w2));
   
    end
end
for w3=2:n3
    if max(abs(x3(:,w3)))~=0;
    x3(:,w3)=(x3(:,w3)-mean((x3(:,w3))))./std(x3(:,w3));
   
    end
end
YtrainSET=T{1:mtrain,3}/mean(T{1:mtrain,3});
YCV=T{mtrain+1:mtrain+mCV,3}/mean(T{mtrain+1:mtrain+mCV,3});
YtestSET=T{mtrain+mCV+1:end,3}/mean(T{mtrain+mCV+1:end,3});

for j=1:length(thetas1)
r=exp(-x1*thetas1);
h1=1./(1+(r));
g1(j)=(1/m)*sum((h1-YtrainSET)'*x1(:,j));
k=k+1;
thetas1=thetas1-(Alpha/m)*x1'*(log(h1)-YtrainSET);
ETrain(k)=-(1/m)*sum((YtrainSET.*log(h1))+(1-YtrainSET).*log(1-h1))+(lamda2/(2*m))*sum(thetas1.^2);

end
k=1;
for j=1:length(thetas)
r=exp(-x2*thetas);
h2=1./(1+(r));
g2(j)=(1/m)*sum((h2-YCV)'*x2(:,j));

thetas=thetas-(Alpha/m)*x2'*(log(h2)-YCV);
Ecv(k)=-(1/m)*sum((YCV.*log(h2))+(1-YCV).*log(1-h2))+(lamda2/(2*m))*sum(thetas.^2);
k=k+1;
end
k=1;
for j=1:length(thetas)
r=exp(-x3*thetas);
h3=1./(1+(r));
g3(j)=(1/m)*sum((h3-YtestSET)'*x3(:,j));

thetas=thetas-(Alpha/m)*x3'*(log(h3)-YtestSET);
ETest(k)=-(1/m)*sum((YtestSET.*log(h3))+(1-YtestSET).*log(1-h3))+(lamda2/(2*m))*sum(thetas.^2);
k=k+1;
end


plot(ETrain,'k')
hold on
plot(Ecv,'b')
hold on
plot(ETest,'r')
legend('Trian','CV','Test')
