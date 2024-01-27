import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import os

from utils.tools import AvgMeter
from utils.lr_sched import adjust_learning_rate
from utils.visualize import save_img

def train_transformer(model, input_generator, train_loader, optimizer, writer, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    for epoch in range(args.start_epoch, args.epochs):

        # train
        model.train()
        ce_loss = AvgMeter()
        cnt_correct = 0
        cnt_total = 0
        for iter, (inputs, labels) in enumerate(train_loader):

            # adjusts learning rate
            adjust_learning_rate(optimizer, iter / len(train_loader) + epoch, args)

            optimizer.zero_grad()
            
            # embedding and mask
            inputs = inputs.to(args.device)

            tokens, _, _ = input_generator.generate_all(inputs, 0)

            tokens_ = tokens.to(args.device)
            labels = labels.to(args.device)

            # forward
            loss, pred_classes = model.forward_class_loss(tokens_, labels) # ce loss
            loss.backward()

            # train的准确率
            cnt_correct += (torch.argmax(pred_classes, dim=1) == labels).sum()
            cnt_total += labels.shape[0] 

            # update statistics
            ce_loss.update(loss.item())
            
            # write every k iters
            if (iter+1) % args.print_every_iter == 0:
                msg = f'[Epoch] {epoch} [iter] {iter+1} [Loss] {loss.item():.4f}({ce_loss.avg():.4f}) ' + \
                    f'[LR] {optimizer.param_groups[-1]["lr"]:.6f}'
                print(msg)

            optimizer.step()
        
        acc = (cnt_correct * 100.) / cnt_total
        print(f'Train Accuray: {acc:.2f}%')
        # write every epoch
        to_write = {
            "Loss": ce_loss.avg(),
            "LR": optimizer.param_groups[-1]["lr"]
        }

        msg = f"[End of Epoch] {epoch} "
        for k, v in to_write.items():
            writer.add_scalar(f"{k}/train", v, epoch)
            msg += f"[{k}] {v:.6f}  "
        print(msg)

        # save checkpoint every k epoch
        if (epoch+1) % args.save_every_epoch == 0 or epoch+1 == args.epochs:
            save_path = f'transformer_ckpt/{args.dataset}/{args.model}_lr{args.lr}_classify'
            os.makedirs(save_path, exist_ok=True)
            save_path += f'/ep{epoch}.pt'
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'args': args,
                    }, save_path)