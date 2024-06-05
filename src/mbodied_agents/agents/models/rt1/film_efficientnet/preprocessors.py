# Subject to the terms and conditions of the Apache License, Version 2.0 that the original code follows, 
# I have retained the following copyright notice written on it.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# You can find the original code from here[https://github.com/google-research/robotics_transformer].



# This prepocessor is a simplfied version of original code.
# This will pad the image and crop it at a random location
# The cropped image maintain original image size and almost full field of view.
# Omitted preprocessor_test.py. But you can do simple test by running this code.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from skimage import data


# images: [B, 3 ,H, W]
# receive images whose values is in the range [0,255]
# -> change images values range into [0.0 ,1.0]
# -> padding and crop image
def convert_dtype_and_crop_images(images: torch.Tensor, ratio: float = 0.07):
    if images.dtype == torch.uint8:
        images = images / 255.
    images = images.to(torch.float32)

    _, _, height, width = images.shape
    ud_pad = int(height * ratio)
    lr_pad = int(width * ratio)
    
    images= F.pad(images, pad=(lr_pad, lr_pad, ud_pad, ud_pad))

    shif_h = torch.randint(0, 2*ud_pad+1, size=[])
    shif_w = torch.randint(0, 2*lr_pad+1, size=[])

    grid_h, grid_w = torch.meshgrid(torch.arange(shif_h, shif_h + height),
                                    torch.arange(shif_w, shif_w + width), 
                                    indexing='ij')
    images = images[..., grid_h, grid_w] # fancy index

    return images



if __name__ == '__main__':
    images = data.coffee() # ndarray
    images = np.tile(np.expand_dims(images,0), (10,1,1,1)) # batch size: 10
    images = torch.from_numpy(images).permute(0, 3, 1, 2) # (b, h, w, c) -> (b, c, h, w)
    images = convert_dtype_and_crop_images(images)
    print(torch.max(images))
    print(torch.min(images))
    print(images.shape)
    image_show = images.permute(0, 2, 3 , 1).numpy()
    plt.imshow(image_show[0])
    plt.show()