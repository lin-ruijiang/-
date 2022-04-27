function output = cascade(classifiers,img,thresh)
result = 0;
px = size(classifiers,1);
weightSum = sum(classifiers(:,12));
for i = 1:px
    classifier = classifiers(i,:);
    haar = classifier(1);
    pixelX = classifier(2);
    pixelY = classifier(3);
    haarX = classifier(4);
    haarY = classifier(5);
 
    haarVal = calcHaarVal(img,haar,pixelX,pixelY,haarX,haarY);
    if haarVal >= classifier(9) && haarVal <= classifier(10)
        score = classifier(12);
    else
        score = 0;
    end
   result = result + score;
end

if result >= weightSum*thresh
    output = 1;
else
    output = 0; 
end
end