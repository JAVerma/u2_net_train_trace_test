#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:37:12 2023

@author: shreyank
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:15:10 2023

@author: shreyank
"""


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from glob import glob
import os.path as osp
import requests
import json
import os 

 
INPUT_DIR= "/home/ai-team/members/shreyank/Transparent_shadow/testing/comprehensive/Bad/*"
OUTPUT_DIR= "/home/ai-team/members/shreyank/Transparent_shadow/testing/comprehensive/rmbg_Bad/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_image(url):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    return Image.open(resp.raw)

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

def process(imagepath,live=False):
    # files = [
    #     ('image_file', (imagepath.split('/')[-1],
    #      open(imagepath, 'rb'), 'image/jpeg'))
    # ]
    payload ={}
    # {    "bg_id": bg_id,}
    url = get_url_from_fpath(imagepath)

    if live:
        request_url = "http://internal-imagewizard-lb-new-1763306184.us-east-1.elb.amazonaws.com/removebg/removebg720"
    else:
        print("url")
        request_url="http://172.16.14.41:6969/removebg720/replacecarbg/"

    response = requests.request("POST", request_url, data={'image_url': url,'resize_custom':False}, files=files)
    if response.status_code == 500:
        print(response.text)
    return download_image(json.loads(response.text)['url'])

def resize(image):
    w, h = image.size
    nh = 1080
    nw = int(w*1080/h)
    image = image.resize((nw,nh))
    return image

def main(path):
    try:
        # im_name = path.split("/")[-1].split('.')[0]
        # if osp.exists(osp.join(OUTPUT_DIR, im_name + '.png')):
        #     return
        # if not RAW:
        result = process(path,live=True)
        # result = result.crop(result.getbbox())
            # result3 = process(path, BG_ID1, live=True, tint=True)

        # raw = Image.open(path)
        # raw = resize(raw)

        # result2 = process(path,BG_ID4,tint=tint_params1)
        # result3 = process(path,BG_ID3,tint=tint_params1)
        # result4 = process(path,BG_ID4,tint=tint_params1)
        # print("t")
        # print(result.width,result.height)
        # print(result2.width,result2.height)
        # print(result3.width,result3.height)
        # print(result4.width,result4.height)        
        # # res = Image.new("RGB", (result2.width,result2.height))
        # # res = Image.new("RGB", (result2.width+result.width,result2.height))
        # res = Image.new("RGB", (result.width+result2.width,result.height+result3.height))
        # res.paste(result,(0,0))
        # res.paste(result2,(result.width,0))
        # res.paste(result3,(result.width,result.hieght))
        # # res.paste(result4,(result3.width,result2.height))
        # # # res.paste(result4,(result.width+raw.width,result.height))
        # # # # res.paste(result3,(result.width*2,0))
        # res.save(osp.join(OUTPUT_DIR, im_name + '.png'))
        result.save(osp.join(OUTPUT_DIR, im_name + '.png'))
    except Exception as E:
        print(E)

futures = []
images = glob(INPUT_DIR)
with ThreadPoolExecutor(5) as exe:
    for image in images:
        futures.append(exe.submit(main,images))
    for future in tqdm(as_completed(futures), total=len(images)):
        pass

