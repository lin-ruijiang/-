function facedetection(img_name)  
% 读取RGB图像  
I = imread(img_name);  
% 转换为灰度图像  
gray = rgb2gray(I);  
% 将图像转化为YCbCr颜色空间  
YCbCr = rgb2ycbcr(I);  
% 获得图像宽度和高度  
heigth = size(gray,1);  
width = size(gray,2);  
% 根据肤色模型将图像二值化  
for i = 1:heigth  
    for j = 1:width  
        Y = YCbCr(i,j,1);  
        Cb = YCbCr(i,j,2);  
        Cr = YCbCr(i,j,3);  
        if(Y < 80)  
            gray(i,j) = 0;  
        else  
            if(skin(Y,Cb,Cr) == 1)  
                gray(i,j) = 255;  
            else  
                gray(i,j) = 0;  
            end  
        end  
    end  
end  
% 二值图像形态学处理  
SE=strel('arbitrary',eye(5));    
%gray = bwmorph(gray,'erode');  
% imopen先腐蚀再膨胀  
gray = imopen(gray,SE);  
% imclose先膨胀再腐蚀  
%gray = imclose(gray,SE);  
imshow(gray);  
% 取出图片中所有包含白色区域的最小矩形  
[L,num] = bwlabel(gray,8);  
STATS = regionprops(L,'BoundingBox');  
% 存放经过筛选以后得到的所有矩形块  
n = 1;  
result = zeros(n,4);  
figure,imshow(I);  
hold on;  
for i = 1:num  
    box = STATS(i).BoundingBox;  
    x = box(1);    %矩形坐标x  
    y = box(2);    %矩形坐标y  
    w = box(3);    %矩形宽度w  
    h = box(4);    %矩形高度h  
    % 宽度和高度的比例  
    ratio = h/w;  
    ux = uint8(x);  
    uy = uint8(y);  
    if ux > 1  
        ux = ux - 1;  
    end  
    if uy > 1  
        uy = uy - 1;  
    end  
    % 可能是人脸区域的矩形应满足以下条件：  
    %   1、高度和宽度必须都大于20，且矩形面积大于400  
    %   2、高度和宽度比率应该在范围（0.6,2）内  
    %   3、函数findeye返回值为1  
    if w < 50 || h < 50 || w*h < 2500  
        continue  
    elseif ratio < 2 && ratio > 0.6 && findeye(gray,ux,uy,w,h) == 1  
        % 记录可能为人脸的矩形区域  
        result(n,:) = [ux uy w h];  
        n = n+1;  
    end  
end  
% 对可能是人脸的区域进行标记  
if  size(result,1) == 1 && result(1,1) > 0  
    rectangle('Position',[result(1,1),result(1,2),result(1,3),result(1,4)],'EdgeColor','r');  
else  
    % 如果满足条件的矩形区域大于1则再根据其他信息进行筛选  
    for m = 1:size(result,1)  
        m1 = result(m,1);  
        m2 = result(m,2);  
        m3 = result(m,3);  
        m4 = result(m,4);  
        % 标记最终的人脸区域  
        if m1 + m3 < width && m2 + m4 < heigth  
            rectangle('Position',[m1,m2,m3,m4],'EdgeColor','r');  
        end  
    end  
end   