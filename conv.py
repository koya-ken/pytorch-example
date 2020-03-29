import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

ts = transforms.ToPILImage()
# plt.imshow(im, cmap='gray')
# plt.show()

totensor = transforms.ToTensor()
toimg = transforms.ToPILImage()

while True:
    im = Image.open('image/dog/dog.jpg')
    im = totensor(im)
    out_sample = 6
    n_row = out_sample / 3
    conv1 = nn.Conv2d(3,6,6,2,2)
    conv2 = nn.Conv2d(6,12,6,2,2)
    conv3 = nn.Conv2d(12,24,6,2,2)
    conv4 = nn.Conv2d(36,24,6,2,2)
    pool = nn.MaxPool2d(3)
    up1 = nn.Upsample(scale_factor=2)
    up2 = nn.Upsample(scale_factor=2)
    im.unsqueeze_(0)

    relu = True
    out1 = F.relu(conv1(im))  if relu else conv1(im)
    out2 = F.relu(conv2(out1)) if relu else conv2(out1)
    out3 = F.relu(conv3(out2)) if relu else conv3(out2)
    out3 = pool(out3)
    out4 = up1(out3)
    out5 = up2(out4)
    conv_image = out5
    # out5 = up2(torch.cat([out2,out4],1))
    # out6 = F.relu(conv4(out5))
    # conv_image = out6
    im = torchvision.utils.make_grid(conv_image[0][:,None],nrow=3)
    plt.imshow(toimg(im), cmap='gray')
    # plt.show()
    plt.pause(.5)