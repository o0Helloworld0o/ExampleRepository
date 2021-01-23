import numpy as np
import torch

from util.print_func import *







a = np.random.rand(3, 4)
pp(a)

b = torch.rand(3, 4)
pp(b)



c = np.random.rand(512, 3)
c = torch.rand(512, 3)
pp_verts(c, 'c')


