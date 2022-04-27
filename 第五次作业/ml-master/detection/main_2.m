faceimg = imread('Dwain_Kyles_0001.jpg');
faceDetector = vision.CascadeObjectDetector();
bbox = step(faceDetector, faceimg);
facebox = insertObjectAnnotation(faceimg,'rectangle',bbox,'Face');
figure;imshow(facebox);