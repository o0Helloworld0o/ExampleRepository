import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets







switch = 3




if switch == 0:
    from util.print_func import pp, pp_verts

    a = np.random.rand(3, 4)
    pp(a)

    b = torch.rand(3, 4)
    pp(b)


    c = np.random.rand(512, 3)
    c = torch.rand(512, 3)
    pp_verts(c, 'c')





elif switch == 1:
    from util.dataset_aug import RandomBlur, RandomFlipWithLabel

    img_dir = 'E:/Dataset_Collection/FFHQ'

    tfm = RandomBlur(0.5)
    dataset = datasets.ImageFolder(img_dir, tfm)

    index = 0
    fig, axes = plt.subplots(3, 5, figsize=(16, 18))
    for k, ax in enumerate(axes.flat):
        X, y = dataset[index]
        ax.imshow(X)
        ax.axis('off')
    plt.show()



    tfm = RandomFlipWithLabel(0.5)
    dataset = datasets.ImageFolder(img_dir)

    index = 0
    fig, axes = plt.subplots(3, 5, figsize=(16, 18))
    for k, ax in enumerate(axes.flat):
        X, y = dataset[index]

        # 一般在__getitem__方法中调用带label的数据增强
        y = 1
        X, y = tfm(X, y)

        ax.imshow(X)
        ax.set_title(f'label={y}')
        ax.axis('off')
    plt.show()





elif switch == 3:
    from util.loss_func import WeightedMSELoss

    out = torch.rand(77, 13)
    target = torch.rand(77, 13)


    w = np.ones(13)
    w[7] = 1.5
    criterion = WeightedMSELoss(w)
    loss = criterion(out, target)
    print('【pytorch】')
    print('loss =', loss.item())



    # 验证
    def numpy_mse(y_pred, y_true, w):
        d = y_pred - y_true
        d = d ** 2
        loss = d * w
        loss = loss.mean()
        return loss

    loss = numpy_mse(out.numpy(), target.numpy(), w)
    print('\n【numpy】')
    print('loss =', loss)





elif switch == 4:
    from util.loss_func import FixedMSELoss

    out = torch.rand(77, 56)
    target = torch.rand(77, 56)

    criterion = FixedMSELoss()
    loss = criterion(out, target)
    print('【pytorch】')
    print('loss =', loss.item())