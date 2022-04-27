function val = calcHaarVal(img,haar,pixelX,pixelY,haarX,haarY)
moveX = haarX-1;
moveY = haarY-1;
if haar == 1 
    white = getCorners(img,pixelX,pixelY,pixelX+moveX,pixelY+floor(moveY/2)); 
    black = getCorners(img,pixelX,pixelY+ceil(moveY/2),pixelX+moveX,pixelY+moveY);
    val = white-black;
elseif haar == 2 
    white = getCorners(img,pixelX,pixelY,pixelX+floor(moveX/2),pixelY+moveY);
    black = getCorners(img,pixelX+ceil(moveX/2),pixelY,pixelX+moveX,pixelY+moveY);
    val = white-black;
elseif haar == 3
    white1 = getCorners(img,pixelX,pixelY,pixelX+moveX,pixelY+floor(moveY/3));
    black = getCorners(img,pixelX,pixelY+ceil(moveY/3),pixelX+moveX,pixelY+floor((moveY)*(2/3)));
    white2 = getCorners(img,pixelX,pixelY+ceil((moveY)*(2/3)),pixelX+moveX,pixelY+moveY);
    val = white1 + white2 - black;
elseif haar == 4
    white1 = getCorners(img,pixelX,pixelY,pixelX+floor(moveX/3),pixelY+moveY);
    black = getCorners(img,pixelX+ceil(moveX/3),pixelY,pixelX+floor((moveX)*(2/3)),pixelY+moveY);
    white2 = getCorners(img,pixelX+ceil((moveX)*(2/3)),pixelY,pixelX+moveX,pixelY+moveY);
    val = white1 + white2 - black;
elseif haar == 5 
    white1 = getCorners(img,pixelX,pixelY,pixelX+floor(moveX/2),pixelY+floor(moveY/2));
    black1 = getCorners(img,pixelX+ceil(moveX/2),pixelY,pixelX+moveX,pixelY+floor(moveY/2));
    black2 = getCorners(img,pixelX,pixelY+ceil(moveY/2),pixelX+floor(moveX/2),pixelY+moveY);
    white2 = getCorners(img,pixelX+ceil(moveX/2),pixelY+ceil(moveY/2),pixelX+moveX,pixelY+moveY);
    val = white1+white2-(black1+black2);
end