import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Callable, Union, Any, TypeVar
import torch.nn.functional as F
from abc import abstractmethod

### VQVAE ver2 ###
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]
    
    def forward_index(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        B, H, W, D = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1)  # [BHW]
        encoding_inds = encoding_inds.view(B, H * W)

        return  encoding_inds
    
    def index2token(self, index: int) -> Tensor:
        encoding_one_hot = torch.zeros(self.K, device='cuda:0')
        encoding_one_hot[index] = 1.

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [D]

        return quantized_latents
    
    def indexes2tokens(self, indexes: Tensor) -> Tensor:
        """
        input: [B, L]
        """
        B, L = indexes.shape
        indexes = indexes.view(-1).unsqueeze(1) # [BHW, 1]
        device = indexes.device
        encoding_one_hot = torch.zeros(indexes.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, indexes, 1)  # [BHW x K]

        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(B, L, -1)  # [B, L, D]
        return quantized_latents

class ResidualLayer(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 stride: int,
                 img_size: int,
                 activation_layer: nn.Module = nn.SiLU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 ):
        super(ResidualLayer, self).__init__()
        
        assert stride == 1 or stride == 2 and img_size % 2 == 0
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            norm_layer([out_channels, img_size//stride, img_size//stride]),
            activation_layer(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer([out_channels, img_size//stride, img_size//stride]),
            activation_layer(),
        )
        
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        res = self.residual(input)
        out = self.net(input)
        return res + out


class ResizeLayer(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.target_size)


class VQVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 encoder_depth: int = 6,
                 decoder_depth: int = 6,
                 beta: float = 0.25,
                 img_size: int = 32,
                 activation_layer: nn.Module = nn.SiLU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        self.hidden_dims = hidden_dims
        img_sizes = [img_size // (2 ** i) for i in range(len(hidden_dims))]

        #-------------------------------------------------------------------------------------------
        # Build Encoder
        modules.append(
            ResidualLayer(in_channels, hidden_dims[0], 1, img_size, activation_layer, norm_layer)
        )

        for i in range(len(hidden_dims)-1):
            modules.append(ResidualLayer(hidden_dims[i], hidden_dims[i+1], 2, img_sizes[i], activation_layer, norm_layer))
            modules.append(ResizeLayer(img_sizes[i+1]))
        
        for _ in range(self.encoder_depth):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1], 1, img_sizes[-1], activation_layer, norm_layer))

        modules.append(
            nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1, stride=1),
        )

        self.encoder = nn.Sequential(*modules)
        #-------------------------------------------------------------------------------------------
        # VQ Layer
        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)
        #-------------------------------------------------------------------------------------------
        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                norm_layer([hidden_dims[-1], img_sizes[-1], img_sizes[-1]]),
                activation_layer()
            )
        )

        for _ in range(self.decoder_depth):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1], 1, img_sizes[-1], activation_layer, norm_layer))

        for i in range(len(hidden_dims)-1, 0, -1):
            modules.append(ResidualLayer(hidden_dims[i], hidden_dims[i-1], 1, img_sizes[i], activation_layer, norm_layer))
            modules.append(ResizeLayer(img_sizes[i-1]))


        modules.append(
            nn.Conv2d(hidden_dims[0], in_channels, kernel_size=3, stride=1, padding=1)
        )

        self.decoder = nn.Sequential(*modules)
        #-------------------------------------------------------------------------------------------

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        encoding = self.encode(input)[0]

        quantized_inputs, vq_loss = self.vq_layer(encoding)

        recons = self.decode(quantized_inputs)
        
        return [recons, input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Recons_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self, num_samples: int, device) -> Tensor:
        raise Warning

    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]