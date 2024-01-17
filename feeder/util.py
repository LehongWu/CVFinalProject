import torch
import torch.nn.functional as F

import torchvision
from torchvision import transforms
import os


'''example
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
'''

def get_transform_from_args(args=None):
    transform_list = []

    # mnist
    if args.dataset == 'mnist':
        transform_list += [
            #transforms.Grayscale(3),
            transforms.ToTensor(), 
            transforms.Normalize(mean = (0.1307,),std = (0.3081,))
        ]
        return transforms.Compose(transform_list)
 
    # cifar & imagenet
    if 'train' in args.mode:
        transform_list += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    return transforms.Compose(transform_list)