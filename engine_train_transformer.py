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

        for iter, (inputs, _) in enumerate(train_loader):

            # adjusts learning rate
            adjust_learning_rate(optimizer, iter / len(train_loader) + epoch, args)

            optimizer.zero_grad()
            
            # embedding and mask
            inputs = inputs.to(args.device)

            mask_ratio = random.random()
            mask_ratio = max(0.1, mask_ratio)
            mask_ratio = min(0.75, mask_ratio) # 保证mask ratio不大不小（太小没有意义，太大影响训练）

            tokens, mask, gt = input_generator.generate_all(inputs, mask_ratio)

            tokens_ = tokens.to(args.device)
            mask_ = mask.to(args.device)
            gt_ = gt.to(args.device)

            # forward
            loss, pred_indexes = model.forward_loss(tokens_, gt_, mask_) # ce loss
            loss.backward()

            # show some results
            if iter == 1:
                print("mask ratio:", mask_ratio)
                N = pred_indexes.shape[0]
                D = 256
                H = 8
                W = 8
                with torch.no_grad():
                    pred_tokens = input_generator.vqvae.vq_layer.indexes2tokens(pred_indexes)
                    pred_tokens[~mask] = tokens_[~mask]

                    latent = pred_tokens.permute(0, 2, 1).contiguous().view(N, D, H, W)
                    recons = input_generator.vqvae.decode(latent)
                
                # 用于可视化训练过程中重建效果的代码
                dir = dir = f'visualize/transformer/reconstruct'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                idx = 0
                for i in range(recons.shape[0]):
                    img = inputs[i]
                    path = dir + f'/{idx}_ori.jpg'
                    img_recons = recons[i]
                    path_recons = dir + f'/{idx}.jpg'
                    save_img(img, path)
                    save_img(img_recons, path_recons)
                    idx += 1

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