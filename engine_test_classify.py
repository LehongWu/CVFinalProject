import torch
import torch.nn.functional as F

from tqdm import tqdm
import os

from utils.tools import AvgMeter
from utils.lr_sched import adjust_learning_rate
from utils.visualize import save_img

@torch.no_grad()
def test_transformer(model, input_generator, test_loader, args):

    model.eval()
    input_generator.eval()
    meter_loss = AvgMeter()
    print('Start testing classification...')

    cnt_correct = 0
    cnt_total = 0
    for iter, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(args.device)

        tokens, _, _ = input_generator.generate_all(inputs, 0)

        tokens_ = tokens.to(args.device)

        preds = model.forward_class(tokens_).cpu()
        
        cnt_correct += (torch.argmax(preds, dim=1) == labels).sum()
        cnt_total += labels.shape[0] 

    acc = (cnt_correct * 100.) / cnt_total
    print(f'Test Accuray: {acc:.2f}%')