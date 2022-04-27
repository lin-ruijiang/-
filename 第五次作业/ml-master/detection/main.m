clear;
for x = 7:7
    stImageFilePath  = ['.\LFW\match pairs\',num2str(x,'%04d'),'\'];
    dirImagePathList = dir(strcat(stImageFilePath,'*.jpg'));        %��ȡ���ļ���������ͼƬ��·�����ַ�����ʽ��
    iImageNum        = length(dirImagePathList);                    %��ȡͼƬ��������
    if iImageNum > 0                                                %��������ͼƬ��������ټ�⣬���������
        for i = 1 : iImageNum
            iSaveNum      = int2str(i);
            stImagePath   = dirImagePathList(i).name;
            mImageCurrent = imread(strcat(stImageFilePath,stImagePath));
            mFaceResult   = face_detect(mImageCurrent);
            figure; imshow(mFaceResult);
        end
    end
end