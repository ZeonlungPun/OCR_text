import cv2
import numpy as np
import meaUtils
import pytesseract,easyocr




#识别放在桌面上的A4文件里的文字
webcam=False
path='E:\opencv\Resources\paper.jpg'

cap=cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)

#A4长度和和宽度mm
scale=1
wP=int(420*scale)
hP=int(596*scale)


while True:
    if webcam:
        success,img=cap.read()
    else:
        img=cv2.imread(path)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    OriImg,findCounters=meaUtils.getContours(img,cThr=[25,75],draw=False,showCanny=False,reverse=False)



    #获取A4纸4个点坐标
    if len(findCounters)!=0:
        biggest=findCounters[0][2]
        imgWrap=meaUtils.warpImg(OriImg,biggest,w=wP,h=hP,pad=20)


        #缩小显示
        #imgWrap = cv2.resize(imgWrap, (0, 0), None, 0.25, 0.25)
        reader=easyocr.Reader(['en'])
        result=reader.readtext(imgWrap)
        print(result)
        text = pytesseract.image_to_string(imgWrap)
        print('text:',text)
        text=text.split('\n')
        text_img=255*np.ones((800,1800),dtype=np.uint8)
        tx,ty=50,50
        for i in text:
            if i!='':
                cv2.putText(text_img,i,(tx,ty),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                ty+=len(i)+50
                if ty>1800:
                    ty=50
                    tx+=50

        cv2.imshow('text',text_img)
        cv2.imshow('wrap', imgWrap)
        key=cv2.waitKey()
        if key==27:
            break



    #
    #cv2.imshow('ori', img)
    #cv2.waitKey()





