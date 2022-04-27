function [faceBound] =  facedetect(img)
img = im2double(img);
img = conv2(img,fspecial('gaussian',3,3),'same');

[m,n] = size(img);


scanItr = 8; 
% 迭代次数
faces = []; 
% compute integral image
intImg = integralImg(img);


load './trainHaar/trainedClassifiers.mat' 

class1 = selectedClassifiers(1:2,:);
class2 = selectedClassifiers(3:12,:);
class3 = selectedClassifiers(13:20,:);
class4 = selectedClassifiers(21:40,:);
class5 = selectedClassifiers(41:70,:);
class6 = selectedClassifiers(71:150,:);
class7 = selectedClassifiers(151:200,:);


for itr = 1:scanItr
    % printout = strcat('Iteration #',int2str(itr),'\n');
    % fprintf(printout);
    for i = 1:2:m-19
        if i + 19 > m 
            break; % 边界检测
        end
        for j = 1:2:n-19
            if j + 19 > n
                break;
            end
            window = intImg(i:i+18,j:j+18); 
            check1 = cascade(class1,window,1);
            if check1 == 1
                check2 = cascade(class2,window,.5);
                if check2 == 1
                    check3 = cascade(class3,window,.5);
                    if check3 == 1
                        check4 = cascade(class4,window,.5);
                        if check4 == 1
                            check5 = cascade(class5,window,.6);
                            if check5 == 1
                                check6 = cascade(class6,window,.6); 
                                if check6 == 1
                                    % fprintf('Passed level 6 cascade.\n');
                                    check7 = cascade(class7,window,.5);
                                    if check7 == 1
                                        % 存储
                                        bounds = [j,i,j+18,i+18,itr];
                                        % fprintf('Face detected!\n');
                                        faces = [faces;bounds];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
   
    tempImg = imresize(img,.8);
    img = tempImg;
    [m,n] = size(img);
    intImg = integralImg(img);
end

if size(faces,1) == 0 
    faceDetector = vision.CascadeObjectDetector();
   faceBound = step(faceDetector, img);
end


if size(faces,1) ~= 0 
   faceBound = zeros(size(faces,1),4);
   maxItr = max(faces(:,5)); 
    for i = 1:size(faces,1)
      if faces(i,5) ~= maxItr
         continue; 
      end
    faceBound(i,:) = floor(faces(i,1:4)*1.25^(faces(i,5)-1));
    end


startRow = 1;
for i = 1:size(faceBound,1)
   if faceBound(i,1) == 0
       startRow = startRow+1; 
   end
end
faceBound = faceBound(startRow:end,:); 

faceBound = [min(faceBound(:,1)),min(faceBound(:,2)),max(faceBound(:,3)),max(faceBound(:,4))];
end

end
