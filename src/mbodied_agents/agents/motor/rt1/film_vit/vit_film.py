import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import EncoderBlock

from .film import FilmLayer

VIT_B16_HIDDEN_SIZE = 768

class VisionTransformerB16(nn.Module):
    def __init__(self, context_dim: int = 512):
        super().__init__()
        net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        film_layers = []

        for layer in net.encoder.layers:
            if isinstance(layer, EncoderBlock):
                mlp_block = layer.mlp
                last_linear_layer = None
                for sublayer in reversed(list(mlp_block.children())):
                    if isinstance(sublayer, nn.Linear):
                        last_linear_layer = sublayer
                        break
    
                film_layer = FilmLayer(
                    last_linear_layer.out_features, 
                    context_dim)
                film_layers.append(film_layer)

        self.film_layers = nn.ModuleList(film_layers)
        self.conv_proj = net.conv_proj
        self.encoder = net.encoder
        

    def forward(self, x, context):

        n, c, h, w = x.shape
        patch_size = 16
        torch._assert(h == 224, f"Wrong image height! Expected 224 but got {h}!")
        torch._assert(w == 224, f"Wrong image width! Expected 224 but got {w}!")
        n_h = h // patch_size
        n_w = w // patch_size

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, VIT_B16_HIDDEN_SIZE, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        film_layers = iter(self.film_layers)

        for encoder_child in self.encoder.children():
            if isinstance(encoder_child, nn.Sequential):
                for layer in encoder_child:
                    x = layer(x)
                    film_layer = next(film_layers, None)
                    if film_layer is not None:
                        x = film_layer(x, context)
            else:
                x = encoder_child(x)

        x = self.encoder.norm(x)
        x = self.heads(x[:, 0])
        return x