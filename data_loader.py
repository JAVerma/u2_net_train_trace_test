# data loader
import numpy as np
from torch.utils.data import Dataset
import cv2,os
from PIL import Image
import torch,cv2
class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform_both = transform[0]
        self.transform_img = transform[1]
        self.transform_mask = transform[2]
    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        if os.path.exists(self.label_name_list[idx]):
            image = cv2.imread(self.image_name_list[idx])
            # image = Image.fromarray(image)
            # bg = Image.new("RGB", image.size)
            # bg.paste(image, (0,0), image)
            image = np.array(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # label = 255 - cv2.imread(self.label_name_list[idx], cv2.IMREAD_GRAYSCALE)
            label=cv2.imread(self.label_name_list[idx])
            label = np.array(label)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            h,w,_=image.shape
        else:
            image = cv2.imread('/home/spyne-4090/members/shreyank/Transparent_shadow/data_2/data/trainA/0a0ce839-d117-4062-b455-d91544112ab7.png')
            # image = Image.fromarray(image)
            # bg = Image.new("RGB", image.size)
            # bg.paste(image, (0,0), image)
            image = np.array(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # label = 255 - cv2.imread(self.label_name_list[idx], cv2.IMREAD_GRAYSCALE)
            label=cv2.imread('/home/spyne-4090/members/shreyank/Transparent_shadow/data_2/data/trainB/0a0ce839-d117-4062-b455-d91544112ab7.png')
            label = np.array(label)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            print('no mask')
            # label=cv2.resize(label,(640,320))
            # label=label/255
            

        # transformed = self.transform(image=image, mask=label)
        label=cv2.resize(label,(image.shape[1],image.shape[0]))
        transf_one = self.transform_both(image=image, image0=label)
        image,mask=transf_one['image'],transf_one['image0']
        # image=image/255
        # mask=mask/255
        image=self.transform_img(image=image)['image']/255
        mask=self.transform_mask(image=mask)['image']/255
        # mask=mask[None]
        transformed={'image':image,'mask':mask}
        # inputs_v=image.permute((1,2,0))
        # cv2.imwrite('/home/spyne-4090/members/Jayant/u2-net/input555.jpg',inputs_v.numpy()*255)
        # transformed['imageB'] = transformed['imageB'][None]/255.
        return transformed
