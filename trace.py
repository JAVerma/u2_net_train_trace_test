import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import cv2
import os

from model.u2net import U2NET
# from model import U2NETP

checkpoint_path = "/home/ai-team/members/Jayant/u2-net/saved_models/u2net/u2net_bce_itr_110_train_-0.04137764844027433_tar_0.34461836381392047.pth"
net = U2NET(3, 3)
net.load_state_dict(torch.load(checkpoint_path))
net.cuda()
net.eval()


transparent_image = cv2.imread("/home/ai-team/members/Jayant/transparent_reflection_data/images_png/0a0f1c88-6316-4e6c-87cd-5c587152d47a.png", cv2.IMREAD_UNCHANGED)
# transparent_image = cv2.cvtColor(transparent_image, cv2.COLOR_BGR2GRAY)
#
print(transparent_image.shape)
w, h = 640, 320
transparent_image = cv2.resize(transparent_image, (w, h))
print(transparent_image.shape)
a = Image.fromarray(transparent_image)
print(a.size)
bg_img = Image.new("RGB", (w, h), (255, 255, 255))
print(bg_img.size)
bg_img.paste(a, (0, 0), a)
# transparent_image = cv2.cvtColor(transparent_image, cv2.COLOR_BGR2GRAY)
# transparent_image = cv2.cvtColor(transparent_image, cv2.COLOR_GRAY2RGB)
transparent_image=np.array(bg_img)
cv2.imwrite('input2.jpg',transparent_image)
#transparent_image=transparent_image[None]
print(transparent_image.shape,'#######################')
data = torch.from_numpy(np.array(transparent_image)).to('cuda')[None]
print(data.shape)
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, data, fp16=True):
        with amp.autocast(enabled=fp16):
            print(data.shape)
            data = data.permute(0, 3, 1, 2).contiguous()
            data=data.contiguous()
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

cv2.imwrite("pretrace2.png",svd_out[0].cpu().numpy())


OUT_PATH = "traced_models1"
os.makedirs(OUT_PATH, exist_ok=True)

with torch.inference_mode(), torch.jit.optimized_execution(True):
    traced_script_module = torch.jit.trace(wrp_model, data)
    # traced_script_module = torch.jit.optimize_for_inference(traced_script_module)

traced_script_module.save(f"{OUT_PATH}/model.pt")
traced_script_module = torch.jit.load(f"{OUT_PATH}/model.pt")
with torch.no_grad():
    o = traced_script_module(data)
o=o[0].cpu().numpy()


# raw=Image.open("/home/ai-team/members/shreyank/Transparent_shadow/shadow_predictions/car_png_/A_26_65.png").convert("RGBA")
# w1,h1=raw.size
# shadow=Image.new('RGB',(w1,h1))
# white=Image.new('RGB',(w1,h1),(255,255,255))

# o=cv2.resize(o,(w1, h1))
# o = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)
# o=Image.fromarray(o)
# o,_,_=o.split()
# white.paste(shadow,(0,0),o)
# white.paste(raw,(0,0),raw)
# print(np.array(raw).shape,w1,h1,'#####################3')
# white.save("postrace2.png")
# cv2.imwrite("postrace.png",white)

