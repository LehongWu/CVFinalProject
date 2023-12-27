import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def save_img(img, path):
    '''
    img: [3, H, W] or [H, W, 3]
    '''
    if isinstance(img, torch.Tensor): # to numpy
        img = img.numpy() 

    assert len(img.shape) == 3

    if img.shape[0] == 3:
        img = np.transpose(img, (1,2,0)) # [3, H, W] => [H, W, 3]

    img_unnorm = np.zeros_like(img)

    for i in range(3):
        img_unnorm[:,:,i] = img[:,:,i] * std[i] + mean[i]

    img_unnorm = img_unnorm * 255
    img_uint8 = np.clip(img_unnorm, 0, 255).astype(np.uint8)

    # create PIL object
    image = Image.fromarray(img_uint8)

    image.save(path)