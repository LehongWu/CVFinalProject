import torch
import torch.nn.functional as F

from tqdm import tqdm
import os

from utils.tools import AvgMeter
from utils.lr_sched import adjust_learning_rate
from utils.visualize import save_img

@torch.no_grad()
def test_tokenizer(model, test_loader, args, MAXNUM=100):

    model.eval()
    meter_loss = AvgMeter()
    print('Start testing tokenizer...')

    idx = 0
    dir = f'visualize/{args.description}'
    os.makedirs(dir)

    for iter, (inputs, _) in enumerate(test_loader):
        
        # forward
        recons = model.reconstruct(inputs)

        loss = F.mse_loss(recons, inputs)
        meter_loss.update(loss.item())

        for i in range(recons.shape[0]):
            img, img_recons = inputs[i], recons[i]
            path, path_recons = dir + f'/{idx}_original.jpg', dir + f'/{idx}_recons.jpg'
            save_img(img, path)
            save_img(img_recons, path_recons)
            idx += 1
    
        if idx > MAXNUM: break

    print(f'Finished. Mean MSE Loss: {meter_loss.avg()}')
