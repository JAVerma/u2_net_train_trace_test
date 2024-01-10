import cv2
import numpy as np 
import os
import shutil
import glob 
from tqdm import tqdm
raw=glob.glob('/home/spyne-4090/members/shreyank/Transparent_shadow/data_2/data/trainA/*')
# mask=glob.glob('/home/spyne-4090/members/shreyank/Transparent_shadow/data_2/data/trainA/*')
print(len(raw))
d=0
for i in tqdm(raw):
    img=cv2.imread(i)
    h,w,_=img.shape
    i2=i.replace('trainA','trainB')
    img1=cv2.imread(i2)
    h1,w1,_=img1.shape
    # print(i,i2)
    # input()
    if h1!=h and w1 !=w:
        print(i)
        print(h,h1,w,w1)
    else:
        d=d+1