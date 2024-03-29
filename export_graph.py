import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import cv2
import os

from model import U2NET
from model import U2NETP

checkpoint_path = "/home/ubuntu/U-2-Net/saved_models/u2net/u2net_bce_itr_20126_train_0.6488732241759726_tar_0.0909561465040751.pth"
net = U2NET(3, 1)
net.load_state_dict(torch.load(checkpoint_path))
net.cuda()
net.eval()


transparent_image = cv2.imread("/home/ubuntu/U-2-Net/jayant.png", cv2.IMREAD_UNCHANGED)
# transparent_image = cv2.cvtColor(transparent_image, cv2.COLOR_BGR2GRAY)
# transparent_image = cv2.cvtColor(transparent_image, cv2.COLOR_GRAY2RGB)

w, h = 320, 320
transparent_image = cv2.resize(transparent_image, (w, h))
a = Image.fromarray(transparent_image)
bg_img = Image.new("RGB", (w, h), (0, 0, 0))
bg_img.paste(a, (0, 0), a)
bg_img = np.array(bg_img)
transparent_image = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
transparent_image = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('input.jpg', transparent_image)
# transparent_image = transparent_image[None]
print(transparent_image.shape, '#######################')
data = torch.from_numpy(np.array(transparent_image)).to('cuda')[None]
print(data.shape)


class WrappedModel(torch.nn.Module):  #forward depend upon transformation done in training
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, data, fp16=True):
        with amp.autocast(enabled=fp16):
            # print(data.shape)
            data = data.permute(0, 3, 1, 2).contiguous()
            data = data.contiguous()
            data = data.div(127.5).sub_(1.0)
            x = self.model(data)[0]
            x = torch.sigmoid(x)
            x = x.mul_(255)
            x = x.to(torch.uint8)
            x = x.permute(0, 2, 3, 1)
            return x.contiguous()


wrp_model = WrappedModel(net).to('cuda').eval()
with torch.no_grad():
    svd_out = wrp_model(data, fp16=False)

print(f'{svd_out[0].shape=}')
cv2.imwrite("pretrace.png", svd_out[0].cpu().numpy())


OUT_PATH = "traced_models"
os.makedirs(OUT_PATH, exist_ok=True)

with torch.inference_mode(), torch.jit.optimized_execution(True):
    traced_script_module = torch.jit.trace(wrp_model, data)
    # traced_script_module = torch.jit.optimize_for_inference(traced_script_module)

traced_script_module.save(f"{OUT_PATH}/model.pt")
traced_script_module = torch.jit.load(f"{OUT_PATH}/model.pt")
with torch.no_grad():
    o = traced_script_module(data)
o = o[0].cpu().numpy()


