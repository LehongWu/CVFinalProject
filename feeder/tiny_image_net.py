import cv2 as cv
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from .util import get_transform_from_args

class TinyImageNet(Dataset):
    def __init__(self, root_dir, split='train', transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir  # data directory
        self.transform = transform
        self.images = []
        target_dir = os.path.join(root_dir, split)
        if split == 'train':
            sub_dirs = [os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
            for sub_dir in sub_dirs:
                for _, _, files in os.walk(sub_dir):
                    for f in files:
                        if f.endswith('.JPEG'):
                            img_dir = os.path.join(sub_dir,'images',f)
                            self.images.append(img_dir)   
        elif split == 'test':
            for _, _, files in os.walk(target_dir):
                    for f in files:
                        if f.endswith('.JPEG'):
                            img_dir = os.path.join(target_dir,'images',f)
                            self.images.append(img_dir)   
        else:
            raise NotImplementedError 

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        return image, 0 

    def __len__(self):
        return len(self.images)
    
def get_tiny_image_net_loader(split='train', args=None):

    transform = get_transform_from_args(args)

    if split == 'train':
        train_dataset = TinyImageNet(root_dir='./datasets/tiny-imagenet-200', split=split, transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
        return train_loader
    
    elif split == 'test':
        test_dataset = TinyImageNet(root_dir='./datasets/tiny-imagenet-200', split=split, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=2)
        return test_loader

    else:
        raise ValueError