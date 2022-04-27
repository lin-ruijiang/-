%清理窗口  
close all  
clear all  
clc  
% 输入图像名字  
img_name = input('请输入图像名字(图像必须为RGB图像，输入0结束)：','s');  
% 当输入0时结束  
while ~strcmp(img_name,'0')  
    % 进行人脸识别  
    facedetection(img_name);  
    img_name = input('请输入图像名字(图像必须为RGB图像，输入0结束)：','s');  
end  