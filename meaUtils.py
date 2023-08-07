import numpy as np
import cv2

def getContours(OriImg,cThr=[100,100],showCanny=False,minArea=0,filter=0,draw=False,reverse=True):
    img=cv2.cvtColor(OriImg,cv2.COLOR_BGR2GRAY)
    img=cv2.GaussianBlur(img,(3,3),1)
    img=cv2.Canny(img,cThr[0],cThr[1])
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img=cv2.dilate(img,kernel,iterations=1)
    img=cv2.erode(img,kernel,iterations=1)
    #要转成黑色背景白色前景，看情况是否需要
    if reverse:
        img=cv2.bitwise_not(img)
    if showCanny:
        cv2.imshow('canny',img)
        cv2.waitKey(0)

    contours,hiearchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    findCounters=[]
    for i in contours:
        area=cv2.contourArea(i)
        if area>minArea:
            peri=cv2.arcLength(i,True)
            #多边形逼近算法
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            bbox=cv2.boundingRect(approx)
            #s是否过滤一些轮廓
            if filter>0:
                if len(approx)==filter:
                    findCounters.append([len(approx),area,approx,bbox,i])
            else:
                findCounters.append([len(approx), area, approx, bbox, i])
    findCounters=sorted(findCounters,key=lambda x:x[1],reverse=True)
    if draw:
        for con in findCounters:
            cv2.drawContours(OriImg,con[4],-1,(0,0,255),3)

    return OriImg,findCounters

#将找出的四个角点排序：左上 1  右上 2 左下：3 右下：4
#通过将X和Y坐标相加减识别出是哪个位置的坐标点
def reorder(myPoints):
    print(myPoints.shape)
    myNewPoints=np.zeros_like(myPoints)
    myPoints=myPoints.reshape((4,2))
    add=myPoints.sum(1)
    myNewPoints[0]=myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]
    return myNewPoints




#利用透视变换
#裁切出对应A4纸张
def warpImg(img,points,w,h,pad=20):
    points=reorder(points)
    #原始点
    pst1=np.float32(points)
    #新点
    pst2=np.float32([[0,0],[w,0],[0,h],[w,h]])
    mat=cv2.getPerspectiveTransform(pst1,pst2)
    imgWrap=cv2.warpPerspective(img,mat,(w,h))
    imgWrap=imgWrap[pad:imgWrap.shape[0]-pad,pad:imgWrap.shape[1]-pad]
    return imgWrap

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2+(pts2[1]-pts1[1])**2)**0.5

