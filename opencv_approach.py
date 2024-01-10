#!/usr/bin/env python
# coding: utf-8

import io

import cv2
import os
import numpy as np
import thinplate.numpy as tps
from google.cloud import vision
from PIL import Image


class TransparentShadow:
    def __init__(self, brightness=0.85):
        cur_file = os.path.dirname(os.path.realpath(__file__))
        self.base_shadow = 255 - np.uint8(cv2.imread(os.path.join(cur_file, "shad_flat_edited.png"), cv2.IMREAD_GRAYSCALE) * brightness)
        self.slant_shadow = 255 - np.uint8(cv2.imread(os.path.join(cur_file, "shad_slant.png"), cv2.IMREAD_GRAYSCALE) * brightness)
        self.vision_api_client = vision.ImageAnnotatorClient()
        self.left = np.array([[57, 236], [57, 648]])
        self.right = np.array([[775, 244], [775, 634]])
        self.padx1 = 250        # can make it dynamic
        self.padx2 = 200
        self.slant_box = cv2.boundingRect(255 - self.slant_shadow)
    
    def filter_ps(self, ps):
        i = 0
        while i < len(ps)-1:
            if abs(ps[i][0] - ps[i+1][0]) < 20:
                avg = [(ps[i][0]+ps[i+1][0])//2, (ps[i][1]+ps[i+1][1])//2]
                ps.pop(i)
                ps.pop(i)
                ps.insert(i,avg)
            i+=1
        return ps
    
    def detect_keypoints(self, transparent_arr):
        image = Image.fromarray(transparent_arr[...,:3])
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')
        content = imgByteArr.getvalue()

        vimg = vision.Image(content=content)
        objects = self.vision_api_client.object_localization(image=vimg).localized_object_annotations

        width, height = image.size
        ps = []
        for o in objects:
            if o.name == 'Wheel' or o.name == 'Tire':
                wheel_cordinates = o.bounding_poly.normalized_vertices
                p1, p2 = wheel_cordinates[2:]
                ps.append([int((p1.x+p2.x)/2*width), int((p1.y+p2.y)/2*height)])

        ps = sorted(ps, key=lambda x: x[0])
        ps = self.filter_ps(ps)
        return ps

    def process_image(self, transparent_arr):
        transparent_arr = cv2.copyMakeBorder(transparent_arr, 0, 0, self.padx1, self.padx2, cv2.BORDER_CONSTANT)
        ps = self.detect_keypoints(transparent_arr)

        x, y, w, h = cv2.boundingRect(transparent_arr[..., 3])
        if len(ps) == 0:
            tx, ty, tw, th = self.slant_box
            shad_crop = self.slant_shadow[ty:ty+th,tx:tx+tw]
            cw,ch = int(w*1.), int(h*1.)
            out = np.ones_like(transparent_arr[...,3]) * 255
            ox = x+w//2 - cw//2
            oy = int(0.14   *h)
            dest = out[y+h+oy-ch:y+h+oy,ox:ox+cw]
            shad_crop = cv2.resize(shad_crop, dest.shape[::-1])
            dest[:] = shad_crop
        else:
            if len(ps) == 1:
                d = transparent_arr.copy()
                d[y:y+int(h*0.45)] = 0
                nx, _, nw, _ = cv2.boundingRect(d[..., 3])
                off = 0.02
                if ps[0][0] > x+w//2:
                    off = 1 - off
                ox = nx + int(off*nw)
                oy = d.shape[0] - np.argmax(d[::-1,ox,3]!=0)
                ps.append([ox,oy])
                ps = sorted(ps, key=lambda x: x[0])
            
            if len(ps) == 3:
                pw = sorted(ps, key=lambda x: x[1])
                ps = [pw.pop(0)]
                ridx = int(abs(ps[0][0]-pw[0][0]) > abs(ps[0][0]-pw[1][0]))
                ps.append(pw.pop(ridx))
            else:
                if ps[0][1] > ps[1][1]:
                    pw = [[x+int(0.08*w), y+int(0.72*h)]]
                    ridx = 0
                else:
                    pw = [[x+int(0.92*w), y+int(0.72*h)]]
                    ridx = 1

            src = np.vstack((self.left, self.left.mean(axis=0)-1, self.right[ridx])).astype(np.float32)
            dst = np.array(ps)
            dst = np.vstack((dst, dst.mean(axis=0)-1, np.array(pw))).astype(np.float32)

            # d = transparent_arr.copy()
            # for i,p in enumerate(dst):
            #     cv2.putText(d, f"{i}", np.int32(p), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255,255), thickness=2)
            # display(Image.fromarray(d))

            src = src/np.array(self.base_shadow.shape[::-1])
            dst = dst/np.array(transparent_arr.shape[-2::-1])

            theta = tps.tps_theta_from_points(src, dst, reduced=False)
            grid = tps.tps_grid(theta, dst, transparent_arr.shape)
            map_x, map_y = tps.tps_grid_to_remap(grid, self.base_shadow.shape)
            tpw = cv2.remap(self.base_shadow, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)

            tx, ty, tw, th = cv2.boundingRect(255-tpw)
            out = np.ones_like(tpw)*255

            cut1 = tpw[:,tx:ps[0][0]]
            cut1 = cv2.resize(cut1, (abs(ps[0][0]-x), cut1.shape[0]))
            out[:,x:ps[0][0]] = cut1

            cut2 = tpw[:,ps[1][0]:tx+tw]
            cut2 = cv2.resize(cut2, (abs(x+w-ps[1][0]), cut2.shape[0]))
            out[:,ps[1][0]:x+w] = cut2

            out[:,ps[0][0]:ps[1][0]] = tpw[:,ps[0][0]:ps[1][0]]

        out = out[:,self.padx1:-self.padx2]
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        transparent_arr = transparent_arr[:,self.padx1:-self.padx2]
        return out, transparent_arr
