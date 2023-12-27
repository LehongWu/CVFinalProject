import torch
import torch.nn as nn

from .vq import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder: define a encoder 

        # decoder: definr a decoder

        # define a vector quantization module
        

    def encoder(self, x):
        # encoding images 
        pass
    
    def decoder(self, x):
        # reconstructing images from quantized_latents 
        pass

    def forward(self, x):
        pass

    def vqencoder(self, x):
        # encoding input images input quantized_latents 
        x = self.encoder(x)
        q_x, _, label = self.vq_layer(x)
        return q_x, label 
    
    def vqdecoder(self, q_x):
        # reconstructing images from quantized_latents 
        return self.decoder(q_x)
