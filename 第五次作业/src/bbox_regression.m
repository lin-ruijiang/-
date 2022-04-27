%{
    dbnames = {'LFPW'};
    imgpathlistfile = strcat('..\', dbnames{1},'\trainset\Path_Images.txt');
    imgpathlist = textread(imgpathlistfile, '%s', 'delimiter', '\n');
    filedir = imgpathlistfile(1:end-15);
    X = zeros(length(imgpathlist), 4);
    Y = zeros(length(imgpathlist), 4);
    M = zeros(1, 4);
    count = 0;
    for i = 1:length(imgpathlist)
        img = im2uint8(imread([filedir,imgpathlist{i}]));
        M = face_detect(img);
        if (M ==[50,100,100,100])
            i
            continue;
        end
        count = count + 1;
        X(count, :) = M;
        shapepath = strcat(imgpathlist{i}(1:end-3), 'pts');
        path = [filedir,shapepath];
        file = fopen(path);

        if ~isempty(strfind(path, 'COFW'))
            shape = textscan(file, '%d16 %d16 %d8', 'HeaderLines', 3, 'CollectOutput', 3);
        else
            shape = textscan(file, '%d16 %d16', 'HeaderLines', 3, 'CollectOutput', 2);
        end
        fclose(file);

        shape_gt = shape{1};
        Y(count,:) = getbbox(shape_gt);
    end
%}
Data = load('X.mat');
X = Data.X;
Data = load('Y.mat');
Y = Data.Y;
[n, ~] = size(X);
for num = 1:n
    M = X(num, :);
    if (M==[0,0,0,0])
        break;
    end
end
num = num - 1;
X1 = X(1:num,:);
Y1 = Y(1:num,:);
MIN = min(X1);
MAX = max(X1);
for i=1:4
    X1(:,i) = (X1(:,i) - MIN(i))/(MAX(i) - MIN(i));
end
O = ones(num, 1);
X1 = [O,X1];
W = inv(X1'*X1)*X1'*Y1;