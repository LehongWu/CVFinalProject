import torch
import torch.nn.functional as F

from tqdm import tqdm
import os

from utils.tools import AvgMeter
from utils.lr_sched import adjust_learning_rate
from utils.visualize import save_img

@torch.no_grad()
def test_transformer(model, vqvae, args, MAXNUM=100):

    model.eval()
    vqvae.eval()
    meter_loss = AvgMeter()
    print('Start testing transformer...')

    idx = 0
    dir = f'visualize/{args.dataset}/generation_mixfeat/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    idx = 0
    for iter in range(MAXNUM):
        print(f"{idx} images generated.")
        if idx >= MAXNUM:
            break
        # initialize
        N = args.bs
        H = 7
        W = 7
        L = H * W 
        D = 64
        T = 8 # 推理轮数
        tokens_map = torch.zeros((N, L, D), dtype=float) # 初始化可以随意，mask_token在infer时加
        mask = torch.zeros((N, L), dtype=bool)
        mask[:, :] = True # 初始全部mask
        labels = torch.zeros(N, dtype=int)
        # 指定label
        # for i in range(10):
        #     labels[i*10:(i+1)*10] = i
        tokens_map = tokens_map.to(args.device).float()
        mask = mask.to(args.device)
        labels = labels.to(args.device)

        for t in range(1, T + 1):
            # foward
            print(f"iter {t}...")
            # tokens_map = tokens_map.float()
            if args.model == 'cond_transformer':
                new_index_map, new_mask = model.infer_one_iter(tokens_map, mask, t, T, labels)
            elif args.model == 'transformer':
                new_index_map, new_mask = model.infer_one_iter(tokens_map, mask, t, T)
            # 迭代更新token和mask
            mask = new_mask
            
            for n in range(N):
                for patch in range(L):
                    index = new_index_map[n, patch]
                    if index >= 0: # 非-1，是新预测的index
                        token = vqvae.vq_layer.index2token(index)
                        tokens_map[n, patch] = token
            # 将tokens转换回image
            latent = tokens_map.permute(0, 2, 1).contiguous().view(N, D, H, W)
            recons = vqvae.decode(latent)

        for i in range(recons.shape[0]):
            img_recons = recons[i]
            path_recons = dir + f'/{idx}.jpg'
            save_img(img_recons, path_recons)
            idx += 1
