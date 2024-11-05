import os
import torch
import shutil
import pathlib
from PIL import Image
from torchvision.utils import make_grid
from torchvision.utils import save_image
from typing import Any, BinaryIO, List, Optional, Tuple, Union


# 将网络输出转为网格图
@torch.no_grad()
def get_grid_image(tensor: Union[torch.Tensor, List[torch.Tensor]], format: Optional[str] = None, is_scale=True, **kwargs,) -> None:
    grid = make_grid(tensor, **kwargs)
    grid = torch.clamp(grid, -1., 1.)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    if is_scale:
        grid = (grid + 1.0) / 2.0
    im = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return im


# 创建文件夹
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

        
# 创建文件夹【用这个】        
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
        
        
# 遍历rootdir文件夹下所有文件和文件夹
# 按照deal_dir和deal_file的方式处理
def iterate_dir(rootdir, deal_dir=None, deal_file=None):
    for dir_or_file in os.listdir(rootdir):
        path = os.path.join(rootdir, dir_or_file)
        if os.path.isfile(path):
            if deal_file == 'pass':
                pass
            elif deal_file:
                deal_file(path)
            else:
                print(path)
        if os.path.isdir(path):
            if deal_dir == 'pass':
                pass
            elif deal_dir:
                deal_dir(path)
            else:
                print(path)


# 移动函数
def movefile(srcfile, dstpath): 
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        # 分离文件名和路径
        fpath, fname = os.path.split(srcfile)  
        if not os.path.exists(dstpath):
            # 创建路径
            os.makedirs(dstpath)  
         # 移动文件
        shutil.move(srcfile, dstpath + fname) 

        
# 加载图片
def pil_load_image(image_path, is_RGB=False):
    assert os.path.exists(image_path)
    image = Image.open(image_path)
    if is_RGB:
        return image.convert('RGB')
    else:
        return image
