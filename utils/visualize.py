import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image

def unnormalize(img):
    '''
    img: [H, W, C]
    '''
    C = img.shape[-1]
    
    if C == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.1307]
        std = [0.3081]
    
    img_unnorm = np.zeros_like(img)
    for i in range(C):
        img_unnorm[:,:,i] = img[:,:,i] * std[i] + mean[i]
    return img_unnorm

def save_img(img, path):
    '''
    img: [C, H, W] or [H, W, C]
    '''
    if isinstance(img, torch.Tensor): # to numpy
        img = img.cpu().numpy() 

    assert len(img.shape) == 3

    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1,2,0)) # [3, H, W] => [H, W, 3]

    img_unnorm = unnormalize(img)

    img_unnorm = img_unnorm * 255
    img_uint8 = np.clip(img_unnorm, 0, 255).astype(np.uint8)

    # create PIL object
    if img.shape[-1] == 3:
        image = Image.fromarray(img_uint8)
    else:
        image = Image.fromarray(img_uint8[...,0], mode='L')

    image.save(path)