# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:28:39 2020

@author: zeng
"""

#coding=utf-8
import os,cv2,numpy
import numpy as np
import pandas as pd
import logging
import math
import copy
from scipy.io import loadmat
logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)s %(levelname)s: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])

def warp_im(img_im, orgi_landmarks,tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    print(M[:2])
    return dst,M[:2]

def draw_landmark(img_im,land):
    

    img = copy.deepcopy(img_im)
    n = np.array(land)
    for i in range(len(land)):
        cv2.circle(img,(int(n[i][0]),int(n[i][1])),2,(0,0,255),-1)
    cv2.imshow('aaa',img)
    #cv2.imwrite('land5.jpg',img)
    #cv2.waitKey(0)

def draw_landmark_warpAffine(img,land,M,cutx,cuty):
    
    new_n = []
    for i in range(len(land)):
        pts = []    
        pts.append(np.squeeze(np.array(M[0]))[0]*(land[i][0]+cutx)+np.squeeze(np.array(M[0]))[1]*(land[i][1]+cuty)+np.squeeze(np.array(M[0]))[2])
        pts.append(np.squeeze(np.array(M[1]))[0]*(land[i][0]+cutx)+np.squeeze(np.array(M[1]))[1]*(land[i][1]+cuty)+np.squeeze(np.array(M[1]))[2])
        new_n.append(pts)
    n = np.array(new_n)    
    '''
    for i in range(len(land)):
        cv2.circle(img,(int(n[i][0]),int(n[i][1])),2,(0,0,255),-1)
        '''
    #cv2.imshow('bbb',img)
    return n
    #cv2.imwrite('land68.jpg',img)
    #cv2.waitKey(0)

def rotate(img,land,cutx,cuty):
    
    #draw_landmark(img,land)
    eyecenter=((land[41][0]+cutx+cutx+land[47][0])*0.5,(land[41][1]+cuty+cuty+land[47][1])*0.5)
    dy=land[47][1]-land[41][1]
    dx=land[47][0]-land[41][0]
    angle=math.atan2(dy,dx)*180.0/math.pi
    M=cv2.getRotationMatrix2D(eyecenter,angle,1)
    dst=cv2.warpAffine(img,M,(img.shape[1], img.shape[0]))
    return M,dst
              
    
    
    

if __name__=='__main__':
    m=loadmat('gt.mat')      #输入68个特征点矩阵
    data=m["Data"]
    path='..\\LFW\\mismatch pairs'          #图片根目录
    j=0;
    for home,dirs,files in os.walk(path):
        for file in files:
            fullname=os.path.join(home,file)
        
    #for i in range(len(m["Data"])):


            matrix=data[j,0]['intermediate_shapes']
            cut=data[j,0]['cut']
            cutx=cut[0,0][0,0]
            cuty=cut[0,0][0,1]
            face_landmarks68=matrix[0,0][0,4]
            img_im=cv2.imread(fullname)
    #pic_path = r'20181216222654763.png'
    #img_im = cv2.imread(pic_path)
            M,dst=rotate(img_im,face_landmarks68,cutx,cuty)
            n=draw_landmark_warpAffine(dst,face_landmarks68,M,cutx,cuty)
    #draw_landmark(img_im,face_landmarks)
    #cv2.imshow('affine_img_im', img_im)
    #dst,M = warp_im(img_im, face_landmarks, coord5point)
    #cv2.imshow('affine', dst)
    #cv2.imwrite('affine.jpg',dst)
    #draw_landmark_warpAffine(dst,face_landmarks68,M)
            minx=int(n[1][0])
            maxx=int(n[1][0])
            miny=int(n[1][1])
            maxy=int(n[1][1])
            for i in range(len(n)):
                if minx>=int(n[i][0]):
                    minx=int(n[i][0])
                if maxx<=int(n[i][0]):
                    maxx=int(n[i][0])
                if miny>=int(n[i][1]):
                    miny=int(n[i][1])
                if maxy<=int(n[i][1]):
                    maxy=int(n[i][1])
            crop_im=dst[miny:maxy,minx:maxx]
    #cv2.imshow('affine_crop_im', crop_im)
            paths='preprocessing\\mismatchpairs\\'      #生成截取的图片目录
            #fullname=os.path.join(home,file)
            filename=str(j)
            new_path=paths+filename+'.jpg'
            cv2.imwrite(new_path,crop_im)
            j=j+1
