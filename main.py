import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets

from util.print_func import pp, pp_verts
from util.dataset_aug import RandomBlur





switch = 1


if switch == 0:
    a = np.random.rand(3, 4)
    pp(a)

    b = torch.rand(3, 4)
    pp(b)


    c = np.random.rand(512, 3)
    c = torch.rand(512, 3)
    pp_verts(c, 'c')



elif switch == 1:
    tfm = RandomBlur(0.5)

    img_dir = 'E:/Dataset_Collection/FFHQ'
    dataset = datasets.ImageFolder(img_dir, tfm)
    print(len(dataset))


    index = 0
    fig, axes = plt.subplots(3, 5, figsize=(16, 18))
    for k, ax in enumerate(axes.flat):
        X, y = dataset[index]
        ax.imshow(X)
        ax.axis('off')
    plt.show()
