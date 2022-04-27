function [X] = print_landmark(Data, x1, x2, x3, x4)
    tmp = Data.img_gray;
    imshow(tmp);
    hold on;
    landmark = Data.intermediate_shapes{5};
    for i = 1:68
       plot(landmark(i,1), landmark(i,2),'y*');
       text(landmark(i,1), landmark(i,2),'\color{yellow}')
    end
    rectangle('Position',[Data.bbox_gt(1),Data.bbox_gt(2),Data.bbox_gt(3),Data.bbox_gt(4)],'EdgeColor','r'); 
    rectangle('Position',[Data.bbox_gt2(1),Data.bbox_gt2(2),Data.bbox_gt2(3),Data.bbox_gt2(4)],'EdgeColor','y');
    rectangle('Position',[x4-Data.cut(1),x1-Data.cut(2),x3-x1,x2-x4],'EdgeColor','b');
    hold off;
end