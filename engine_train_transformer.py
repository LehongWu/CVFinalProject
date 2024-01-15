import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import os

from utils.tools import AvgMeter
from utils.lr_sched import adjust_learning_rate

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

        for iter, (inputs, _) in enumerate(train_loader):

            # adjusts learning rate
            adjust_learning_rate(optimizer, iter / len(train_loader) + epoch, args)

            optimizer.zero_grad()
            
            # embedding and mask
            inputs = inputs.to(args.device)
            mask_ratio = random.random()
            tokens, mask, gt = input_generator.generate_all(inputs, mask_ratio)
            tokens = tokens.to(args.device)
            mask = mask.to(args.device)
            gt = gt.to(args.device)
            # forward
            loss = model.forward_loss(tokens, gt, mask) # ce loss
            
            loss.backward()

            # update statistics
            ce_loss.update(loss.item())
            
            # write every k iters
            if (iter+1) % args.print_every_iter == 0:
                msg = f'[Epoch] {epoch} [iter] {iter+1} [Loss] {loss.item():.4f}({ce_loss.avg():.4f}) ' + \
                    f'[LR] {optimizer.param_groups[-1]["lr"]:.6f}'
                print(msg)

            optimizer.step()
        
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
            save_path = f'transformer_ckpt/{args.model}_lr{args.lr}_{args.description}'
            os.makedirs(save_path, exist_ok=True)
            save_path += f'/ep{epoch}.pt'
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'args': args,
                    }, save_path)