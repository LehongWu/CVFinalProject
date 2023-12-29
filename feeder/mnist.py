import torch
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from .util import get_transform_from_args

# For every dataset, the API is function get_{dataset_name}_loader

def get_mnist_loader(split='train', args=None):

    transform = get_transform_from_args(args)

    if split == 'train':
        train_dataset = torchvision.datasets.MNIST(root='./datasets/mnist', train=True, download=True, transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
        return train_loader
    
    elif split == 'test':
        test_dataset = torchvision.datasets.MNIST(root='./datasets/mnist', train=False, download=True, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=2)
        return test_loader

    else:
        raise ValueError
