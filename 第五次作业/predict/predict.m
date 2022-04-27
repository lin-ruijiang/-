function predict(y)
test_x=y;
[m,~]=size(test_x);
II=ones(m,1);
test_xto1=[II,test_x];
a=min(test_xto1(:,2));
b=max(test_xto1(:,2));
if a-b==0
    test_xto1(:,2)=0;
else
    for j=1:m
        test_xto1(j,2)=(test_xto1(j,2)-a)/(b-a);
    end
end
w=load('30000rounds.mat');
w=w.w;
predict=ones(m,1);
for i=1:m
    f=test_xto1(i,:)*w;
    p=1/(1+exp(f));
    if(p>0.5)
        predict(i)=0;
    else
        predict(i)=1;
    end
end
fp=fopen('result.txt','w');
fprintf(fp,'%d\r\n',predict);
fclose(fp);