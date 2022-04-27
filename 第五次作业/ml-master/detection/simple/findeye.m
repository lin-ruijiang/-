% 判断二值图像中是否含有可能是眼睛的块  
%   bImage----二值图像  
%   x---------矩形左上角顶点X坐标  
%   y---------矩形左上角顶点Y坐标  
%   w---------矩形宽度  
%   h---------矩形长度  
% 如果有则返回值eye等于1，否则为0  
function eye = findeye(bImage,x,y,w,h)  
% 根据矩形相关属性得到二值图像中矩形区域中的数据  
% 存放矩形区域二值图像信息  
part = zeros(h,w);  
% 二值化  
for i = y:(y+h)  
    for j = x:(x+w)  
        if bImage(i,j) == 0  
            part(i-y+1,j-x+1) = 255;  
        else  
            part(i-y+1,j-x+1) = 0;  
        end  
    end  
end  
[L,num] = bwlabel(part,8);  
% 如果区域中有两个以上的矩形则认为有眼睛  
if num < 2  
    eye = 0;  
else  
    eye = 1;  
end   