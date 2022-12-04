import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os
import findcarinfo

plt.style.use('dark_background')

class Car:
    def __init__(self, cardate, cartime, carnum):
        self.__cardate = cardate
        self.__cartime = cartime
        self.__carnum = carnum

    def getcardate(self):
        return self.__cardate
    def getcartime(self):
        return self.__cartime
    def getcarnum(self):
        return self.__carnum
    def setcardate(self, newcardate):
        self.__carnum = newcardate
    def setcartime(self, newcartime):
        self.__carnum = newcartime
    def setcarnum(self, newcarnum):
        self.__carnum = newcarnum
    

#carlist saves car objects
carlist=[]

#function that makes car object from picture

    

#path:이미지 들어있는 경로    
path = 'C:/Users/eric7/Desktop/수업/OOPproject4/OOPproject4'
img_list=[]
possible_img_extension = ['.jpg','.jpeg','.JPG','.bmp','.png', '.PNG']
for(root,dirs,files) in os.walk(path):
    dirs[:] = [dir for dir in dirs if dir!="res"]
    if len(files)>0:
        for file_name in files:
            if os.path.splitext(file_name)[1] in possible_img_extension:
                
                img_list.append(file_name)

for img in img_list:
    carname, cardate, cartime = findcarinfo.findcarinfo(img)
    car = Car(carname, cardate, cartime)
    carlist.append(car)
#print(carlist[2].getcardate())
