import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime

from model.VQVAE4transformer import VQVAE
from model.transformer import InputGenerator, BidirectionalTransformer, ConditionBidirectionalTransformer
from feeder.cifar10 import get_cifar10_loader
from feeder.tiny_image_net import get_tiny_image_net_loader
from feeder.mnist import get_mnist_loader
from engine_train_classify import train_transformer
from engine_test_classify import test_transformer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--mode', type=str, default='train_transformer')
    parser.add_argument('--description', type=str, default='exp')
    parser.add_argument('--print_every_iter', type=int, default=40)
    parser.add_argument('--save_every_epoch', type=int, default=5)
    # data
    parser.add_argument('--dataset', type=str, default='mnist')
    # model
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt', type=str, default='./transformer_ckpt\mnist/transformer_lr0.0003_classify\ep39.pt')
    # training
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--min_lr_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=128)

    args = parser.parse_args()
    print('args', args)
    
    # get model
    if args.model == 'transformer':
        vae_path = './assets\VQVAE-mnist-depth1-ep99.pt'
        input_generator = InputGenerator(
            vae_path=vae_path, 
            in_channels=1, embedding_dim=64, num_embeddings=256,
            hidden_dims=[16, 32, 64], img_size=28,
            encoder_depth=1, decoder_depth=1,
        )
        input_generator.requires_grad_(False)
        model = BidirectionalTransformer(
            num_patches=49, 
            num_embeds=256, 
            embed_dim=64,
            num_class=10,
            depth=8, 
            mlp_ratio=4,
            num_heads=8, 
            norm_layer=nn.LayerNorm,
        )
        vqvae = input_generator.vqvae
        # will be replaced by using config (yaml or py) 
    else:
        raise NotImplementedError
    
    # load checkpoint (resume training or evaluation)
    print("load vqvae from " + vae_path + " to initialize input_generator.")
    checkpoint = torch.load(vae_path, map_location='cpu')
    checkpoint_model = checkpoint['model_state_dict']
    msg = input_generator.vqvae.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    if args.resume or ('test' in args.mode):
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        print("Loading pre-trained checkpoint from: %s" % args.ckpt)
        checkpoint_model = checkpoint['model_state_dict']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    # device
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    print(f'Using Device {args.device}')
    model = model.to(args.device)
    input_generator = input_generator.to(args.device)
    vqvae = vqvae.to(args.device)
    
    # get dataset
    if args.dataset == 'cifar10':
        train_loader = get_cifar10_loader(split='train', args=args)
        test_loader = get_cifar10_loader(split='test', args=args)
    elif args.dataset == 'tiny_image_net':
        train_loader = get_tiny_image_net_loader(split='train', args=args)
        test_loader = get_tiny_image_net_loader(split='test', args=args)
    elif args.dataset == 'mnist':
        train_loader = get_mnist_loader(split='train', args=args)
        test_loader = get_mnist_loader(split='test', args=args)
    else:
        raise NotImplementedError
    
    # get optimizer
    if args.mode in ['train_transformer']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    # get writer
    args.description = f'{args.mode}_' + args.description + f'_{datetime.now().strftime("%Y-%m-%d-%H")}'
    if args.mode in ['train_transformer']:
        writer = SummaryWriter(f'exp/{args.description}')

    # train / test
    if args.mode == 'train_transformer':
        train_transformer(model, input_generator, train_loader, optimizer, writer, args)
    elif args.mode == 'test_transformer':
        test_transformer(model, input_generator, test_loader, args)
    else:
        raise NotImplementedError
    
    print('Finished.')