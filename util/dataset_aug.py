import numpy as np
from PIL import Image




class RandomBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        assert(isinstance(img, Image.Image))
        if np.random.rand() >= self.p:
            return img

        down_scale = np.random.choice([2, 3, 4], 1, p=[1/3]*3)
        old_size = img.size[0]
        new_size = old_size // down_scale
        img = img.resize([new_size, new_size])
        img = img.resize([old_size, old_size])
        return img



