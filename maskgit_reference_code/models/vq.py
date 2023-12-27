import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    '''
    Reference:
    Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).
    '''
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        # num_embeddings: the number of embeddings of codebook
        # embedding_dim: the dimensions of each embedding
        # beta: the weight of 'embedding loss'
        pass

    def forward(self, latents: torch.Tensor) :
        # latents: features from encoder

        # Compute L2 distance between latents and embedding weights
        # Get the encoding indices that has the min distance
        # Quantize the latents

        # Compute the VQ Losses
            # commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
            # embedding_loss = F.mse_loss(quantized_latents, latents.detach())
            # vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        # quantized_latents = latents + (quantized_latents - latents).detach()

        # return quantized_latents, vq_loss, encoding indices
        pass 
    