import torch
import torch.nn.functional as F

from tqdm import tqdm
import os

from utils.tools import AvgMeter
from utils.lr_sched import adjust_learning_rate

def train_tokenizer(model, train_loader, optimizer, writer, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''

    for epoch in range(args.start_epoch, args.epochs):

        # train
        model.train()
        meter_loss, meter_recons_loss, meter_vq_loss = AvgMeter(), AvgMeter(), AvgMeter()

        for iter, (inputs, _) in enumerate(train_loader):

            # adjusts learning rate
            adjust_learning_rate(optimizer, iter / len(train_loader) + epoch, args)

            optimizer.zero_grad()
            
            # forward
            inputs = inputs.to(args.device)
            outputs = model(inputs) # [recons, input, vq_loss]
    
            # get loss
            results = model.loss_function(*outputs)
            
            loss, recons_loss, vq_loss = results['loss'], results['Recons_Loss'], results['VQ_Loss'] 
            
            loss.backward()

            # update statistics
            meter_loss.update(loss.item())
            meter_recons_loss.update(recons_loss.item())
            meter_vq_loss.update(vq_loss.item())
            
            # write every k iters
            if (iter+1) % args.print_every_iter == 0:
                msg = f'[Epoch] {epoch} [iter] {iter+1} [Loss] {loss.item():.4f}({meter_loss.avg():.4f}) ' + \
                    f'[Recons_Loss] {recons_loss.item():.4f}({meter_recons_loss.avg():.4f}) ' + \
                    f'[VQ_Loss] {vq_loss.item():.4f}({meter_vq_loss.avg():.4f}) ' + \
                    f'[LR] {optimizer.param_groups[-1]["lr"]:.6f}'
                print(msg)

            optimizer.step()
        
        # write every epoch
        to_write = {
            "Loss": meter_loss.avg(),
            "Recons_Loss": meter_recons_loss.avg(),
            "VQ_Loss": meter_vq_loss.avg(),
            "LR": optimizer.param_groups[-1]["lr"]
        }

        msg = f"[End of Epoch] {epoch} "
        for k, v in to_write.items():
            writer.add_scalar(f"{k}/train", v, epoch)
            msg += f"[{k}] {v:.6f}  "
        print(msg)

        # save checkpoint every k epoch
        if (epoch+1) % args.save_every_epoch == 0 or epoch+1 == args.epochs:
            save_path = f'ckpt/{args.model}_lr{args.lr}_{args.description}'
            os.makedirs(save_path, exist_ok=True)
            save_path += f'/ep{epoch}.pt'
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'args': args,
                    }, save_path)