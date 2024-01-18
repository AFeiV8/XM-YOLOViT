import cv2
import math
import tqdm
import random

import torch
import torch.nn as nn
from torchvision import transforms

import os.path
from PIL import Image


# Estimating atmospheric light values using Dark Channel Prior
class DarkChannelPrior(nn.Module):
    def __init__(self, kernel_size, top_sample_ratio, open_threshold=True):
        super().__init__()

        # dark channel prior
        self.kernel_size = kernel_size
        self.pad = nn.ReflectionPad2d(padding=kernel_size // 2)  # (c,h+2p,w+2p)
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), padding=0)  # (c*kh*kw,n)

        # estimate A
        self.top_sample_ratio = top_sample_ratio
        self.open_threshold = open_threshold

    def forward(self, image):
        # dark channel
        b, c, h, w = image.shape
        image_pad = self.pad(image)
        local_patches = self.unfold(image_pad)
        dc, _ = torch.min(local_patches, dim=1, keepdim=True)

        # estimate A
        top_sample_nums = int(h * w * self.top_sample_ratio)
        searchidx = torch.argsort(dc, dim=-1, descending=True)[:, :, :top_sample_nums]
        searchidx = searchidx.repeat(1, 3, 1)
        image_ravel = image.view(b, 3, -1)
        value = torch.gather(image_ravel, dim=2, index=searchidx)
        airlight, _ = torch.max(value, dim=-1, keepdim=True)
        airlight = airlight.squeeze(-1)
        if self.open_threshold:
            airlight = torch.clamp(airlight, max=0.89)

        A = airlight.sum() / b / 3
        return A


# Generating Foggy Images Using Atmospheric Scattering Models
def Fogging(name, save_path, A, s):
    img = cv2.imread(name)
    img_f = img / 255.0
    (row, col, chs) = img.shape

    if s == 'train':
        beta = 0.078
    else:
        beta = random.uniform(0.07, 0.105)
    size = 55
    center = (row // 3, col // 2)
    for j in range(row):
        for l in range(col):
            d = -0.05 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            tx = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * tx + A * (1 - tx)
    cv2.imwrite(f'{save_path}', img_f * 255)


device = torch.device('cpu')

sampled_path = ''
img_path = os.listdir(sampled_path)

data_path = ''
save_path = ''
names = os.listdir(data_path)
n = len(names)
cls_data = ''  # S, M, L, Rand
s = ''  # train, val, test

imgs = list()
for i in img_path:
    img = Image.open(os.path.join(sampled_path, i))
    transform = transforms.Compose([transforms.Resize([1080, 1920]), transforms.ToTensor()])
    img = transform(img)
    imgs.append(img)
imgs = torch.stack(imgs, dim=0).to(device)

# Estimating atmospheric light values using Dark Channel Prior
dkp = DarkChannelPrior(7, 0.009).to(device)
A = dkp(imgs).numpy()
print(f'The atmospheric light value A is:\n{A}\n')

# Generating Foggy Images Using Atmospheric Scattering Models
for name, i in zip(tqdm.tqdm(names), range(n)):
    name = os.path.join(data_path, name)
    save = os.path.join(cls_data, s, name)
    Fogging(name, save, A, s)
