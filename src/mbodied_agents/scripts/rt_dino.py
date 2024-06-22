from timm import models, list_models, list_modules
import torch
from x_transformers import ViTransformerWrapper, TransformerWrapper, Encoder, Decoder
# m = list_models('vit_large_patch14_reg4_dinov2')[-1]
encoder = models.create_model('vit_large_patch14_reg4_dinov2', pretrained = True, features_only = True)

# print(help(encoder))


# encoder = ViTransformerWrapper(
#     image_size = 256,
#     patch_size = 32,
#     attn_layers = Encoder(
#         dim = 512,
#         depth = 6,
#         heads = 8
#     )
# )

from PIL import Image

# Open an image file
with Image.open('path_to_your_image.jpg') as img:
    # Resize the image
    img_resized = img.resize((512, 512))
    
    # Save the resized image
    img_resized.save('path_to_your_resized_image.jpg')

decoder = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        cross_attend = True,
    )
)

img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

encoded = encoder(img)
decoder(caption, context = encoded) # (1, 1024, 20000)
