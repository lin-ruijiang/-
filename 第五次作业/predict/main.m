clear;
y=ones(1600,1);                             %�������ݼ���С
for x = 1:1600
    stImageFilePath  = ['.\preprocessing\mismatchpairs\'];        %��������Ŀ¼    
    img1=[];
    img2=[];
    stImagePath = [num2str(2*x-2),'.jpg'];
    img1 = imread(strcat(stImageFilePath,stImagePath));
    img1 = rgb2gray(img1);
    stImagePath = [num2str(2*x-1),'.jpg'];
    img2=  imread(strcat(stImageFilePath,stImagePath));
    img2= rgb2gray(img2);
    dis  = compare(img1,img2);
    y(x)=dis;
end
predict(y);