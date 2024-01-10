#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:03:07 2023

@author: shreyank
"""

from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
# root_path="/home/shreyank/spyne/transparent_shadow/testing/Clearmarket_processed/"
# pred_path="/home/shreyank/spyne/transparent_shadow/testing/Clearmarket_pred_new/"
# save_path="/home/shreyank/spyne/transparent_shadow/testing/Clearmarket_new_pred_updated/"
# root_path="/home/shreyank/spyne/transparent_shadow/testing/Carwago_processed/"
# pred_path="/home/shreyank/spyne/transparent_shadow/testing/Carwago_pred_new/"
# save_path="/home/shreyank/spyne/transparent_shadow/testing/Carwago_new_pred_updated/"

# root_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/level2/check_A/"
# pred_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/level2/check_B_jpg/"
# save_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/level2/pred/"



# root_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/test_300323/test_processed/"
root_path="/home/ai-team/members/shreyank/Transparent_shadow/testing/comprehensive/rmbg_Bad/"
pred_path="/home/ai-team/members/shreyank/Transparent_shadow/testing/comprehensive/pred_bad/"
save_path="/home/ai-team/members/shreyank/Transparent_shadow/testing/comprehensive/final_output2/bad/"
# root_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/check_A/"
# pred_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/pred_car_png/"
# save_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/prediction_on_train/"
os.makedirs(save_path,exist_ok=True)
images=os.listdir(root_path)
for i in tqdm(range(len(images))):

    raw=Image.open(root_path+images[i])#.convert("RGBA")
    w1,h1=raw.size
    shadow=Image.new('RGB',(w1,h1))
    white=Image.new('RGB',(w1,h1),(255,255,255))
    prediction=cv2.imread(pred_path+images[i])
    # prediction=255-prediction
    prediction=cv2.resize(prediction,(w1, h1))
    prediction= cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
    prediction= cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
    prediction=Image.fromarray(prediction)
    prediction,_,_=prediction.split()
    white.paste(shadow,(-4,-4),prediction)
    # white.save("1postrace.png")
    white.paste(raw,(0,0),raw)
    # print(np.array(raw).shape,w1,h1,'#####################3')
    white.save(save_path+images[i])
    # cv2.imwrite("postrace.png",white)

# from PIL import Image
# import cv2
# import numpy as np
# import os
# root_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/check_A/"
# pred_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/pred_car_png/"
# save_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/prediction_on_train/"
# images=["0a000fc5-bb12-492d-b5f1-4934da64445a.png"]
# os.makedirs(save_path,exist_ok=True)
# # images=os.listdir(root_path)
# for i in range(len(images)):

#     raw=Image.open("/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/check_A/0a000fc5-bb12-492d-b5f1-4934da64445a.png")#.convert("RGBA")
#     w1,h1=raw.size
#     shadow=Image.new('RGB',(w1,h1))
#     white=Image.new('RGB',(w1,h1),(255,255,255))
#     prediction=cv2.imread("/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/pred_car_png/0a000fc5-bb12-492d-b5f1-4934da64445a.png")
#     prediction=255-prediction
#     prediction=cv2.resize(prediction,(w1, h1))
#     prediction= cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
#     prediction= cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
#     prediction=Image.fromarray(prediction)
#     prediction,_,_=prediction.split()
#     white.paste(shadow,(0,0),prediction)
#     # white.save("1postrace.png")
#     white.paste(raw,(0,0),raw)
#     # print(np.array(raw).shape,w1,h1,'#####################3')
#     white.save(save_path+images[i])
#     # cv2.imwrite("postrace.png",white)





# from PIL import Image
# import cv2
# import numpy as np
# import os
# root_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/check_A/"
# pred_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/pred_car_png/"
# save_path="/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/prediction_on_train/"
# images=["0a000fc5-bb12-492d-b5f1-4934da64445a.png"]
# os.makedirs(save_path,exist_ok=True)
# # images=os.listdir(root_path)
# for i in range(len(images)):

#     raw=Image.open("/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/check_A/A_5_170.png")#.convert("RGBA")
#     w1,h1=raw.size
#     shadow=Image.new('RGB',(w1,h1))
#     white=Image.new('RGB',(w1,h1),(255,255,255))
#     prediction=cv2.imread("/home/shreyank/spyne/transparent_shadow/Transparent-Shadow/pred_car_png/A_12_43.png")
#     prediction=255-prediction
#     prediction=cv2.resize(prediction,(w1, h1))
#     prediction= cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
#     prediction= cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
#     prediction=Image.fromarray(prediction)
#     prediction,_,_=prediction.split()
#     white.paste(shadow,(0,0),prediction)
#     # white.save("1postrace.png")
#     white.paste(raw,(0,0),raw)
#     # print(np.array(raw).shape,w1,h1,'#####################3')
#     white.save(save_path+images[i])
    
#     # cv2.imwrite("postrace.png",white)
