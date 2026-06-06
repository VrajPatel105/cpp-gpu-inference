# Please note that this and other following examples were taking from book:
# "programming massively parallel processors: a hands-on approach 4th edition"


url = "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg?cs=srgb&dl=pexels-pixabay-104827.jpg&fm=jpg"

import torch, os, math, gzip, pickle
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from pathlib import Path

from torch import tensor
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
from torch.utils.cpp_extension import load_inline

img_dir = Path(r"C:\My Projects\Deep Learning Projects\cpp-gpu-inference\3. gpu-fundamentals-pmpp\images")
path_img = img_dir / "cat.jpg"

img = io.read_image(str(path_img))
print(img.shape)   # torch.Size([3, 3560, 5360])
img[:2, :3, :4]

def show_img(x, figsize=(4,3), save_path=None, **kwargs):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape) == 3:
        x = x.permute(1, 2, 0)   # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

img2 = tvf.resize(img, 150, antialias=True)
ch, h, w = img2.shape
ch, h, w, h*w

show_img(img2, save_path=img_dir / "cat_resized.png")

def rgb2grey_py(x):
    c, h, w = x.shape
    n = h * w
    x = x.flatten()
    res = torch.empty(n, dtype=torch.float32, device=x.device)
    for i in range(n):
        res[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n]
    return res.view(h, w)

img_g = rgb2grey_py(img2)

show_img(img_g, cmap='gray', save_path=img_dir / "cat_gray.png")