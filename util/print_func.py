import numpy as np
import torch





def pp(x, prefix=None):
    if isinstance(x, np.ndarray):
        pp_array(x, prefix)
    elif isinstance(x, torch.Tensor):
        pp_tensor(x, prefix)



def pp_array(x, prefix=None):
    min_val, max_val = x.min(), x.max()
    mean_val, std_val = x.mean(), x.std()
    string = ''
    if prefix is not None:
        string += f'【{prefix}】 '
    string += '{}, {}, [{:.3f}, {:.3f}], mean={:.3f}, std={:.3f}'.format(
        x.shape, x.dtype, min_val, max_val, mean_val, std_val)
    print(string)



def pp_tensor(x, prefix=None):
    min_val, max_val = x.min().item(), x.max().item()
    mean_val, std_val = x.mean().item(), x.std().item()
    string = ''
    if prefix is not None:
        string += f'【{prefix}】 '
    string += '{}, {}, [{:.3f}, {:.3f}], mean={:.3f}, std={:.3f}'.format(
        list(x.size()), x.dtype, min_val, max_val, mean_val, std_val)
    print(string)




def pp_verts(x, prefix=None):
    if isinstance(x, np.ndarray):
        pp_array(x, prefix)
    elif isinstance(x, torch.Tensor):
        pp_tensor(x, prefix)
    for c in range(3):
        min_val, max_val = x[:, c].min(), x[:, c].max()
        span = max_val - min_val
        print('c={}, [{:.3f}, {:.3f}], span={:.3f}'.format(c, min_val, max_val, span))




