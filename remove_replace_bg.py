#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:29:47 2023

@author: shreyank
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:53:16 2022

@author: shreyank
"""

import io
import imutils
import json
import math
import mimetypes
import os
from pathlib import Path
from pprint import pprint
import shutil

import boto3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image


# remove_bg_url = "http://172.16.17.152:7079/removebg720/replacecarbg/"
remove_bg_url="http://internal-imagewizard-lb-new-1763306184.us-east-1.elb.amazonaws.com/removebg/removebg720"
replace_bg_url="http://internal-imagewizard-lb-new-1763306184.us-east-1.elb.amazonaws.com/removebg720/replacecarbg/"

s3 = boto3.resource(
    "s3",
    aws_access_key_id="AKIATXZFWTWQY3A7PQH4",
    aws_secret_access_key="ln+GEFMpVL+//bQl2VTzVellaay6SIN4VKbNFcD7",
)


def save_to_cloud(
    img,
    imagename,
    bucketname="spyne",
    key="AI/app/edited/",
    process="tyre_detection",
    extension=".png",
):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    aws_path = key + process + "_" + imagename + extension
    content_type = mimetypes.guess_type(aws_path)[0]
    s3.Bucket(bucketname).upload_fileobj(
        io.BytesIO(img_byte_arr), aws_path, ExtraArgs={"ContentType": content_type}
    )
    url = f"https://{bucketname}.s3.amazonaws.com/{aws_path}"
    return url


def get_url_from_fpath(fpath):
    url = save_to_cloud(img=Image.open(fpath), imagename=os.path.basename(fpath)[:-4])
    return url


def download_from_url(url, save_fpath):
    out = requests.get(url=url, stream=True)
    with open(save_fpath, "wb") as f:
        out.raw.decode_content = True
        shutil.copyfileobj(out.raw, f)

root_dir="/home/ai-team/members/shreyank/Transparent_shadow/testing/comprehensive/Bad/"


# mask_dir="/home/shreyank/Downloads/3 Sixty/mask"
images=os.listdir(root_dir)
# masks=os.listdir(mask_dir)
# save_no_bg="/home/shreyank/spyne/transparent_shadow/testing/no_bg_Carwago/"
# os.makedirs(save_no_bg,exist_ok=True)
save_final_out="/home/ai-team/members/shreyank/Transparent_shadow/testing/comprehensive/rmbg_Bad/"
os.makedirs(save_final_out,exist_ok=True)
# wall_mask="/home/shreyank/Downloads/3 Sixty/ref.jpg"
for i in range(len(images)):
    print(i)
    image=images[i]
    fpath=os.path.join(root_dir,image)
    # mask=os.path.join(mask_dir,image)
#fpath = (os.path.join(root_dir, 'data', 'test', 'test.jpg'))
    url = get_url_from_fpath(fpath)
        
    resp = requests.post(url=remove_bg_url, data={'image_url': url,'resize_custom':False})
    # resp = requests.post(url=remove_bg_url, data={'image_url': url,'bg_id':130})
    output_url = json.loads(resp.text)["url"]
    image_name_new=image.split(".")[0]+".png"
    save_path=os.path.join(save_final_out,image_name_new)
    download_from_url(output_url,save_path)
    # no_bg_image=cv2.imread(save_path, cv2.IMREAD_UNCHANGED)
    # mask_img=cv2.imread(mask,0)
    # #mask_img=cv2.resize(mask_img,[np.shape(no_bg_image)[1],np.shape(no_bg_image)[0]])
    # img=no_bg_image[:,:,:3]
    # car_mask=no_bg_image[:,:,3]
    # final_mask=np.logical_or(car_mask,mask_img).astype("uint8")
    # final_image=np.zeros([np.shape(mask_img)[0],np.shape(mask_img)[1],3])
    # wall=cv2.imread(wall_mask)
    # wall=cv2.resize(wall,np.shape(final_mask.T))
    # final_image[final_mask>0]=img[final_mask>0]
    # final_image[final_mask==0]=wall[final_mask==0]
    # final_save_path=os.path.join(save_final_out,image)
    # cv2.imwrite(final_save_path,final_image)