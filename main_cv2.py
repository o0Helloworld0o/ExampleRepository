import numpy as np
import matplotlib.pyplot as plt

from util.cv2_func import add_text





switch = 0

if switch == 0:
    img = (np.ones([256, 256, 3]) * 255).astype(np.uint8)

    text = '中文示例'
    img = add_text(img, text, (30, 10))
    plt.imshow(img)
    plt.show()




