#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:20:57 2023

@author: shreyank
"""

import glob,os,cv2, numpy as np
import zlib
from PIL import Image, ImageFilter
# '/home/ai-team/Desktop/transparent_shadow/training/train_A/-176_df141a20_Exterior_1.png'
z=glob.glob('/home/ai-team/members/shreyank/Transparent_shadow/data_dark_140423/data_dark/train_A/*.png')
print(len(z))
z1=glob.glob('/home/ai-team/members/shreyank/Transparent_shadow/data_dark_140423/data_dark/train_B/*.png')
# z=glob.glob('/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/a/*.png')
# z1=glob.glob('/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/b/*.png')
# /home/shreyank/spyne/transparent_shadow/testing/Carwago_no_bg
# print(len(z),len(z1))
c=0
import concurrent.futures

def crop(i):
    print(i)
    traina=Image.open(i)
    traina.save('0.png')
    trainb=Image.open(i.replace('_A','_B')).convert('RGBA')
    mask=Image.open(i.replace('train_A/*.png','processed/mask/*.png'))
    mask=Image.fromarray(255-np.array(mask))
    x,y,x1,y1=mask.getbbox()
    print(x,y,y1,x1)
    w,h=traina.size
    if x>=40:
        if w-x1>=40:
            if h-y1>=140:
                traina=traina.crop((x-40,y-10,x1+40,y1+140))
                trainb=trainb.crop((x-40,y-10,x1+40,y1+140))
            else:
                dif=140-h+y1
                print(y-10,'checkcheckchek')
             

                traina=traina.crop((x-40,y-10,x1+40,h))
                trainb=trainb.crop((x-40,y-10,x1+40,h))
                
                traina=np.array(traina)
                trainb=np.array(trainb)
                traina=cv2.copyMakeBorder(traina,0,dif,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                trainb=cv2.copyMakeBorder(trainb,0,dif,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                traina=Image.fromarray(traina)
                trainb=Image.fromarray(trainb)
        else:
            difx=40-w+x1
            traina=np.array(traina)
            trainb=np.array(trainb)
            traina=cv2.copyMakeBorder(traina,0,0,0,difx,cv2.BORDER_CONSTANT,value=(0,0,0))
            trainb=cv2.copyMakeBorder(trainb,0,0,0,difx,cv2.BORDER_CONSTANT,value=(0,0,0))
            traina=Image.fromarray(traina)
            trainb=Image.fromarray(trainb)
            w,_=traina.size
            if h-y1>=140:
                traina=traina.crop((x-40,y-10,w,y1+140))
                trainb=trainb.crop((x-40,y-10,w,y1+140))
            else:
                dif=140-h+y1
                traina=traina.crop((x-40,y-10,w,h))
                trainb=trainb.crop((x-40,y-10,w,h))
                traina=np.array(traina)
                trainb=np.array(trainb)
                traina=cv2.copyMakeBorder(traina,0,dif,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                trainb=cv2.copyMakeBorder(trainb,0,dif,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                traina=Image.fromarray(traina)
                trainb=Image.fromarray(trainb)
    else:
        difx1=40-x
        traina=np.array(traina)
        trainb=np.array(trainb)
        traina=cv2.copyMakeBorder(traina,0,0,difx1,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        trainb=cv2.copyMakeBorder(trainb,0,0,difx1,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        traina=Image.fromarray(traina)
        trainb=Image.fromarray(trainb)
        if w-x1>=40:
            if h-y1>=140:
                traina=traina.crop((0,y-10,x1+40,y1+140))
                trainb=trainb.crop((0,y-10,x1+40,y1+140))
            else:
                dif=140-h+y1
                traina=traina.crop((0,y-10,x1+40,h))
                trainb=trainb.crop((0,y-10,x1+40,h))
                traina=np.array(traina)
                trainb=np.array(trainb)
                traina=cv2.copyMakeBorder(traina,0,dif,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                trainb=cv2.copyMakeBorder(trainb,0,dif,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                traina=Image.fromarray(traina)
                trainb=Image.fromarray(trainb)
        else:
            difx=40-w+x1
            traina=np.array(traina)
            trainb=np.array(trainb)
            traina=cv2.copyMakeBorder(traina,0,0,0,difx,cv2.BORDER_CONSTANT,value=(0,0,0))
            trainb=cv2.copyMakeBorder(trainb,0,0,0,difx,cv2.BORDER_CONSTANT,value=(0,0,0))
            traina=Image.fromarray(traina)
            trainb=Image.fromarray(trainb)
            w,_=traina.size
            if h-y1>=140:

                traina=traina.crop((0,y-10,w,y1+140))
                trainb=trainb.crop((0,y-10,w,y1+140))
            else:

                dif=140-h+y1

                traina=traina.crop((0,y-10,w,h))
                trainb=trainb.crop((0,y-10,w,h))
                traina=np.array(traina)
                trainb=np.array(trainb)
                traina=cv2.copyMakeBorder(traina,0,dif,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                trainb=cv2.copyMakeBorder(trainb,0,dif,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                traina=Image.fromarray(traina)
                trainb=Image.fromarray(trainb)    
    os.makedirs('/home/ai-team/members/shreyank/Transparent_shadow/data_dark_140423/data_dark/check_A',exist_ok=True)
    os.makedirs('/home/ai-team/members/shreyank/Transparent_shadow/data_dark_140423/data_dark/check_B',exist_ok=True) 
    name=os.path.basename(i)
    print(traina.size,trainb.size,name)
    traina.save(os.path.join('/home/ai-team/members/shreyank/Transparent_shadow/data_dark_140423/data_dark/check_A',name))
    trainb.save(os.path.join('/home/ai-team/members/shreyank/Transparent_shadow/data_dark_140423/data_dark/check_B',name))
  
def paster(i):
    a=Image.open(i)
    b=Image.open(i.replace('_A','_B')).convert('RGBA')
    shadow=Image.new('RGB',b.size)
    b=Image.fromarray(255-np.array(b))
    b,_,_=b.split()
    a.paste(shadow,(0,0),b)
    a.save(i)
    print('################')
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executer:
    executer.map(crop,z)
    # print(i)
    # img=cv2.imread(i,cv2.IMREAD_UNCHANGED)
    # img=255-img
    # img=Image.open(i)
#     # i1=i.replace('.jpg','.png')
#     # os.remove(i)
    # bg=Image.new('RGB',img.size,(255,255,255))
    # bg.paste(img,(0,0),img)
    # bg.save(i)
    # ii=i.replace('train_B','train_A')
    # if os.path.exists(ii):
    #     c=c+1
    # else:
    #     os.remove(i)
    #     print('##################')
# print(c)
