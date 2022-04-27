function [facedet] = face_detect(faceimg)
faceDetector = vision.CascadeObjectDetector();
bbox = step(faceDetector, faceimg);
[M, ~] = size(bbox);
max = 0;
index = 0;
for i = 1:M
    if (bbox(i,3)*bbox(i,4)>max)
        max = bbox(i,3)*bbox(i,4);
        index = i;
    end
end
[w, h] = size(faceimg);
if (index==0)
   bbox = [50,100,100,100];
   index = 1;
   facedet = bbox;
   return;
end
recFace.x          = bbox(index,1);
recFace.y          = bbox(index,2);
recFace.width      = bbox(index,3);
recFace.height     = bbox(index,4);
%recFace.y         = min(recFace.y + recFace.height * 0.25, w-recFace.height);

ptFaceCenter.x     = recFace.x + recFace.width / 2;
ptFaceCenter.y     = recFace.y + recFace.height / 2;

recFace.x         = round(ptFaceCenter.x - recFace.width * 0.4);
recFace.y         = round(ptFaceCenter.y - recFace.height * 0.35);
recFace.width     = round(recFace.width * 0.8) ;
recFace.height    = round(recFace.height) ;


facedet = [round(recFace.x), round(recFace.y), round(recFace.width), round(recFace.height)];
end