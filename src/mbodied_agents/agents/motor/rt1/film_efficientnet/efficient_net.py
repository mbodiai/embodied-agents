from torch import nn
from torchvision.models import EfficientNet_B5_Weights, efficientnet_b5
from torchvision.models.efficientnet import MBConv

from .film import FilmLayer


class EfficientNetB5(nn.Module):
    def __init__(self, context_dim: int = 512):
        super().__init__()
        net = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        film_layers = []
        
        for layer in net.features:
          for sublayer in layer:
            if isinstance(sublayer, MBConv):
              film_layers.append(FilmLayer(sublayer.out_channels, context_dim))
        
        # Don't add a film layer to the last layer
        self.film_layers = nn.ModuleList(film_layers[:-1])
        self.features = net.features

    def forward(self, x, context):
      film_layers = iter(self.film_layers)
      film_layer = next(film_layers, None)

      for layer in self.features:
        for sublayer in layer:
          x = sublayer(x)
          if isinstance(sublayer, MBConv):
              if film_layer is not None:
                x = film_layer(x, context)
                film_layer = next(film_layers, None)
              else:
                return x
      return x

# import torchinfo
# from torchinfo import summary
# model = EfficientNetB3()
# summary(model, input_size=[(6, 3, 300, 300),(6, 512)])