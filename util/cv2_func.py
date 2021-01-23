import numpy as np
from PIL import Image, ImageDraw, ImageFont






def add_text(img, text, point=(10, 10), text_color=(0, 250, 0), text_size=30):
    assert(isinstance(img, np.ndarray))
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font_path = 'C:/Windows/Fonts/simsun.ttc'   # 宋体 常规
    fontText = ImageFont.truetype(font_path, text_size, encoding='utf-8')
    draw.text(point, text, text_color, font=fontText)
    img = np.asarray(img)
    return img



