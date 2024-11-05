import os
import cv2
import glob
import json
import lmdb
import torch
import numpy as np
import torch.utils.data as data
from io import BytesIO
from PIL import Image
from torchvision import transforms
from auxiliary.pseudos import PseudoModel
from SCDataset.SCDataset import *
import torch
from PIL import Image
from diffdata.preprocessing import PreProcess

DEFAULT_AFAD_root = '../datasets/AFAD/AFAD-Full/'
area_index_dict = {}
area_index_dict['background'] = 0
area_index_dict['face'] = 1
area_index_dict['left-eyebrown'] = 2
area_index_dict['right-eyebrown'] = 3
area_index_dict['left-eye'] = 4
area_index_dict['right-eye'] = 5
area_index_dict['nose'] = 6
area_index_dict['upper-lip'] = 7
area_index_dict['teeth'] = 8
area_index_dict['under-lip'] = 9
area_index_dict['hair'] = 10
area_index_dict['left-ear'] = 11
area_index_dict['right-ear'] = 12
area_index_dict['neck'] = 13


class MT_ControlNET_V3(data.Dataset):
    def __init__(self, root='../../datasets/MT-Dataset/', fp16=False, dim=(256,256)):
        super(MT_ControlNET_V3, self).__init__()
        self.fp16 = fp16
        self.root = root
        self.makeup_names = glob.glob('%s/*.png' %(os.path.join(self.root, 'images', 'makeup')))
        self.non_makeup_names = glob.glob('%s/*.png' %(os.path.join(self.root, 'images', 'non-makeup'))) 
        self.data = self.makeup_names + self.non_makeup_names
        self.dim = dim
        
    def __len__(self):
        return len(self.data)
    
    def get_color_gray_img(self, image_gray, image_colr, image_mask):
        selected_keys = ['background', 'hair', 'neck']
        selected_mask = image_mask == area_index_dict[selected_keys[0]]
        for key in selected_keys:
            selected_mask += image_mask == area_index_dict[key]
        color_gray_img = image_colr * selected_mask + image_gray * (1 - selected_mask)
        return color_gray_img

    def __getitem__(self, idx):
        filename = self.data[idx]
        image_gray = cv2.imread(filename, 0)
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB)
        image_colr = cv2.imread(filename)
        image_colr = cv2.cvtColor(image_colr, cv2.COLOR_BGR2RGB)
        image_mask = cv2.imread(filename.replace('images', 'segs'))
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2RGB)
        
        if self.dim:
            # INTER_LINEAR INTER_AREA INTER_NEAREST INTER_CUBIC INTER_LANCZOS4
            image_gray = cv2.resize(image_gray, self.dim, interpolation = cv2.INTER_CUBIC)
            image_colr = cv2.resize(image_colr, self.dim, interpolation = cv2.INTER_CUBIC)
            image_mask = cv2.resize(image_mask, self.dim, interpolation = cv2.INTER_CUBIC)

        if self.fp16:
            source = image_colr.astype(np.float16) / 255.0
            image_colr = image_colr.astype(np.float16) / 255.0
            image_gray = image_gray.astype(np.float16) / 255.0
            # image_colr = (image_colr.astype(np.float16) / 127.5) - 1.0
            # image_gray = (image_gray.astype(np.float16) / 127.5) - 1.0
        else:
            source = image_colr.astype(np.float32) / 255.0
            image_colr = image_colr.astype(np.float32) / 255.0
            image_gray = image_gray.astype(np.float32) / 255.0
            # image_colr = (image_colr.astype(np.float32) / 127.5) - 1.0
            # image_gray = (image_gray.astype(np.float16) / 127.5) - 1.0
        # target
        color_gray_img = self.get_color_gray_img(image_gray, image_colr, image_mask)
        color_gray_img = color_gray_img / 0.5 - 1
        # target(jpg):-1, 1 source(hint): (0,1) 
        return dict(jpg=color_gray_img, txt='makeup style transfer', hint=source, path=filename)

    
class MT_Dataset_V3(data.Dataset):
    def __init__(self, root=None, dim=(256, 256), t0=100, inv_steps=40, dataset_len=10):
        super(MT_Dataset_V3, self).__init__()
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        self.dim = dim
        with open(os.path.join(self.root, 'makeup.txt'), 'r') as f:
            self.makeup_names = [name.strip() for name in f.readlines()]
        with open(os.path.join(self.root, 'non-makeup.txt'), 'r') as f:
            self.non_makeup_names = [name.strip() for name in f.readlines()]
        if dataset_len > 0:
            self.makeup_names.sort()
            self.non_makeup_names.sort()
            makeup_len = min(dataset_len, len(self.makeup_names))
            non_makeup_len = min(dataset_len, len(self.non_makeup_names))
            self.makeup_names = self.makeup_names[:makeup_len]
            self.non_makeup_names = self.non_makeup_names[:non_makeup_len]
        self.inv_dir = os.path.join(self.root, 'inv_%d_%d' %(t0, inv_steps))

    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def __getitem__(self, idx):
        # src_img, src_inv, ref_img, ref_inv
        prompt = 'makeup style transfer'
        idx_s = torch.randint(0, len(self.non_makeup_names), (1, )).item()
        idx_r = torch.randint(0, len(self.makeup_names), (1, )).item()
        src_img_filename = self.non_makeup_names[idx_s]
        ref_img_filename = self.makeup_names[idx_r]
        src_img_basename = os.path.basename(src_img_filename)
        ref_img_basename = os.path.basename(ref_img_filename)
        src_inv_filename = os.path.join(self.inv_dir, '%s.pth' %(src_img_basename))
        ref_inv_filename = os.path.join(self.inv_dir, '%s.pth' %(ref_img_basename))
        
        src_img = cv2.imread(os.path.join(self.root, 'images', src_img_filename))
        ref_img = cv2.imread(os.path.join(self.root, 'images', ref_img_filename))
        src_msk = cv2.imread(os.path.join(self.root, 'segs', src_img_filename))
        ref_msk = cv2.imread(os.path.join(self.root, 'segs', ref_img_filename))
        src_inv = torch.load(os.path.join(self.root, src_inv_filename), map_location=torch.device('cpu'))
        ref_inv = torch.load(os.path.join(self.root, ref_inv_filename), map_location=torch.device('cpu'))
        
        # Do not forget that OpenCV read images in BGR order.
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        src_msk = cv2.cvtColor(src_msk, cv2.COLOR_BGR2RGB)
        ref_msk = cv2.cvtColor(ref_msk, cv2.COLOR_BGR2RGB)
        src_inv = src_inv.squeeze(0)
        ref_inv = ref_inv.squeeze(0)
        
        if self.dim:
            src_img = cv2.resize(src_img, self.dim, interpolation=cv2.INTER_AREA)
            ref_img = cv2.resize(ref_img, self.dim, interpolation=cv2.INTER_AREA)
            src_msk = cv2.resize(src_msk, self.dim, interpolation=cv2.INTER_AREA)
            ref_msk = cv2.resize(ref_msk, self.dim, interpolation=cv2.INTER_AREA)

        src_img = src_img.astype(np.float32) / 255.0
        ref_img = ref_img.astype(np.float32) / 255.0
        return dict(src_img=src_img, src_inv=src_inv, ref_img=ref_img, ref_inv=ref_inv, src_msk=src_msk, ref_msk=ref_msk, txt=prompt)


# add pseudo ground truth
class MT_Dataset_V4(data.Dataset):
    def __init__(self, root=None, dim=(256, 256), t0=100, inv_steps=40, dataset_len=10, gpu_id=3):
        super(MT_Dataset_V4, self).__init__()
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        self.dim = dim
        with open(os.path.join(self.root, 'makeup.txt'), 'r') as f:
            self.makeup_names = [name.strip() for name in f.readlines()]
        with open(os.path.join(self.root, 'non-makeup.txt'), 'r') as f:
            self.non_makeup_names = [name.strip() for name in f.readlines()]
        if dataset_len > 0:
            self.makeup_names.sort()
            self.non_makeup_names.sort()
            makeup_len = min(dataset_len, len(self.makeup_names))
            non_makeup_len = min(dataset_len, len(self.non_makeup_names))
            self.makeup_names = self.makeup_names[:makeup_len]
            self.non_makeup_names = self.non_makeup_names[:non_makeup_len]
        self.inv_dir = os.path.join(self.root, 'inv_%d_%d' % (t0, inv_steps))
        self.pseudo_model = PseudoModel(gpu_id)

    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def __getitem__(self, idx):
        # src_img, src_inv, ref_img, ref_inv
        prompt = 'makeup style transfer'
        idx_s = torch.randint(0, len(self.non_makeup_names), (1,)).item()
        idx_r = torch.randint(0, len(self.makeup_names), (1,)).item()
        src_img_filename = self.non_makeup_names[idx_s]
        ref_img_filename = self.makeup_names[idx_r]
        src_img_basename = os.path.basename(src_img_filename)
        ref_img_basename = os.path.basename(ref_img_filename)
        src_inv_filename = os.path.join(self.inv_dir, '%s.pth' % (src_img_basename))
        ref_inv_filename = os.path.join(self.inv_dir, '%s.pth' % (ref_img_basename))

        src_img = cv2.imread(os.path.join(self.root, 'images', src_img_filename))
        ref_img = cv2.imread(os.path.join(self.root, 'images', ref_img_filename))
        src_msk = cv2.imread(os.path.join(self.root, 'segs', src_img_filename))
        ref_msk = cv2.imread(os.path.join(self.root, 'segs', ref_img_filename))
        src_inv = torch.load(os.path.join(self.root, src_inv_filename), map_location=torch.device('cpu'))
        ref_inv = torch.load(os.path.join(self.root, ref_inv_filename), map_location=torch.device('cpu'))

        # Do not forget that OpenCV read images in BGR order.
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        src_msk = cv2.cvtColor(src_msk, cv2.COLOR_BGR2RGB)
        ref_msk = cv2.cvtColor(ref_msk, cv2.COLOR_BGR2RGB)
        src_inv = src_inv.squeeze(0)
        ref_inv = ref_inv.squeeze(0)

        if self.dim:
            src_img = cv2.resize(src_img, self.dim, interpolation=cv2.INTER_AREA)
            ref_img = cv2.resize(ref_img, self.dim, interpolation=cv2.INTER_AREA)
            src_msk = cv2.resize(src_msk, self.dim, interpolation=cv2.INTER_AREA)
            ref_msk = cv2.resize(ref_msk, self.dim, interpolation=cv2.INTER_AREA)

        src_img = src_img.astype(np.float32) / 255.0
        ref_img = ref_img.astype(np.float32) / 255.0
        PGT_SR, PGT_RS = self.pseudo_model.generate_pseudo_GT(os.path.join(self.root, 'images', src_img_filename),
                                                              os.path.join(self.root, 'images', ref_img_filename))
        return dict(src_img=src_img, src_inv=src_inv, ref_img=ref_img, ref_inv=ref_inv, src_msk=src_msk,
                    ref_msk=ref_msk, pgt_sr=PGT_SR, pgt_rs=PGT_RS, txt=prompt)


class Fill50k(data.Dataset):
    def __init__(self, root='../../../../datasets/fill50k', fp16=False, dim=(256, 256)):
        self.root = root
        self.fp16 = fp16
        self.data = []
        self.dim = dim
        json_path = os.path.join(self.root, 'prompt.json')
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.root, source_filename))
        target = cv2.imread(os.path.join(self.root, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if self.dim:
            source = cv2.resize(source, self.dim, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, self.dim, interpolation=cv2.INTER_AREA)

        if self.fp16:
            source = source.astype(np.float16) / 255.0
            target = (target.astype(np.float16) / 127.5) - 1.0
        else:
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, ref_img=source, src_img=source)


class MT_Dataset_DoubleControl(data.Dataset):
    def __init__(self, root=None, dim=(256, 256), gpu_id=None):
        super(MT_Dataset_DoubleControl, self).__init__()
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        self.dim = dim
        with open(os.path.join(self.root, 'makeup.txt'), 'r') as f:
            self.makeup_names = [name.strip() for name in f.readlines()]
        with open(os.path.join(self.root, 'non-makeup.txt'), 'r') as f:
            self.non_makeup_names = [name.strip() for name in f.readlines()]
        self.pseudo_model = PseudoModel(gpu_id)

    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def __getitem__(self, idx):
        # src_img, src_inv, ref_img, ref_inv
        prompt = 'makeup style transfer'
        idx_s = torch.randint(0, len(self.non_makeup_names), (1,)).item()
        idx_r = torch.randint(0, len(self.makeup_names), (1,)).item()
        src_img_filename = self.non_makeup_names[idx_s]
        ref_img_filename = self.makeup_names[idx_r]
        # src_img_basename = os.path.basename(src_img_filename)
        # ref_img_basename = os.path.basename(ref_img_filename)
        # src_inv_filename = os.path.join(self.inv_dir, '%s.pth' % (src_img_basename))
        # ref_inv_filename = os.path.join(self.inv_dir, '%s.pth' % (ref_img_basename))

        src_img = cv2.imread(os.path.join(self.root, 'images', src_img_filename))
        ref_img = cv2.imread(os.path.join(self.root, 'images', ref_img_filename))
        # src_msk = cv2.imread(os.path.join(self.root, 'segs', src_img_filename))
        # ref_msk = cv2.imread(os.path.join(self.root, 'segs', ref_img_filename))
        # src_inv = torch.load(os.path.join(self.root, src_inv_filename), map_location=torch.device('cpu'))
        # ref_inv = torch.load(os.path.join(self.root, ref_inv_filename), map_location=torch.device('cpu'))

        # Do not forget that OpenCV read images in BGR order.
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        # src_msk = cv2.cvtColor(src_msk, cv2.COLOR_BGR2RGB)
        # ref_msk = cv2.cvtColor(ref_msk, cv2.COLOR_BGR2RGB)
        # src_inv = src_inv.squeeze(0)
        # ref_inv = ref_inv.squeeze(0)

        if self.dim:
            src_img = cv2.resize(src_img, self.dim, interpolation=cv2.INTER_AREA)
            ref_img = cv2.resize(ref_img, self.dim, interpolation=cv2.INTER_AREA)
            # src_msk = cv2.resize(src_msk, self.dim, interpolation=cv2.INTER_AREA)
            # ref_msk = cv2.resize(ref_msk, self.dim, interpolation=cv2.INTER_AREA)

        src_img = src_img.astype(np.float32) / 255.0
        ref_img = ref_img.astype(np.float32) / 255.0
        PGT_SR, PGT_RS = self.pseudo_model.generate_pseudo_GT(os.path.join(self.root, 'images', src_img_filename),
                                                              os.path.join(self.root, 'images', ref_img_filename))
        # return dict(src_img=src_img, src_inv=src_inv, ref_img=ref_img, ref_inv=ref_inv, src_msk=src_msk,
        #             ref_msk=ref_msk, pgt_sr=PGT_SR, pgt_rs=PGT_RS, txt=prompt)
        return dict(jpg=(PGT_SR-0.5)*2.0, txt=prompt, ref_img=ref_img, src_img=src_img)


# add makeup
class MT_ControlNET_AddMakeUp(data.Dataset):
    def __init__(self, root=None, dim=(256, 256)):
        super(MT_ControlNET_AddMakeUp, self).__init__()
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        self.makeup_names = glob.glob('%s/*.png' % (os.path.join(self.root, 'images', 'makeup')))
        self.non_makeup_names = glob.glob('%s/*.png' % (os.path.join(self.root, 'images', 'non-makeup')))
        self.data = self.makeup_names + self.non_makeup_names
        self.dim = dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]
        basename = os.path.basename(filename)
        if "non-makeup" in filename:
            txt = 'non-makeup person'
        else:
            txt = 'makeup person'

        image_colr = cv2.imread(filename)
        image_colr = cv2.cvtColor(image_colr, cv2.COLOR_BGR2RGB)
        image_gray = cv2.imread(os.path.join(self.root, 'gray_images', basename))
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB)

        if self.dim:
            image_gray = cv2.resize(image_gray, self.dim, interpolation=cv2.INTER_CUBIC)
            image_colr = cv2.resize(image_colr, self.dim, interpolation=cv2.INTER_CUBIC)

        image_gray = image_gray.astype(np.float32) / 255.0
        image_colr = (image_colr.astype(np.float32) / 127.5) - 1.0
        return dict(jpg=image_colr, txt=txt, ref_img=image_gray, path=filename)


class Fill50k_AddColor(data.Dataset):
    def __init__(self, root='../../../../datasets/fill50k', fp16=False, dim=(256, 256)):
        self.root = root
        self.fp16 = fp16
        self.data = []
        self.dim = dim
        json_path = os.path.join(self.root, 'prompt.json')
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        prompt = 'add color'

        source = cv2.imread(os.path.join(self.root, target_filename), 0)
        target = cv2.imread(os.path.join(self.root, target_filename))

        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if self.dim:
            source = cv2.resize(source, self.dim, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, self.dim, interpolation=cv2.INTER_AREA)

        if self.fp16:
            source = source.astype(np.float16) / 255.0
            target = (target.astype(np.float16) / 127.5) - 1.0
        else:
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, ref_img=source, src_img=source)


class Fill50k_BinaryAddColor(data.Dataset):
    def __init__(self, root='../../../../datasets/fill50k', fp16=False, dim=(256, 256)):
        self.root = root
        self.fp16 = fp16
        self.data = []
        self.dim = dim
        json_path = os.path.join(self.root, 'prompt.json')
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        prompt = 'add color'

        source = cv2.imread(os.path.join(self.root, source_filename), 0)
        target = cv2.imread(os.path.join(self.root, target_filename))

        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if self.dim:
            source = cv2.resize(source, self.dim, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, self.dim, interpolation=cv2.INTER_AREA)

        if self.fp16:
            source = source.astype(np.float16) / 255.0
            target = (target.astype(np.float16) / 127.5) - 1.0
        else:
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, ref_img=source, src_img=source)


class Fill50k_Reconstruct(data.Dataset):
    def __init__(self, root='../../../../datasets/fill50k', fp16=False, dim=(256, 256)):
        self.root = root
        self.fp16 = fp16
        self.data = []
        self.dim = dim
        json_path = os.path.join(self.root, 'prompt.json')
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        prompt = 'None'

        source = cv2.imread(os.path.join(self.root, target_filename))
        target = cv2.imread(os.path.join(self.root, target_filename))

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if self.dim:
            source = cv2.resize(source, self.dim, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, self.dim, interpolation=cv2.INTER_AREA)

        if self.fp16:
            source = source.astype(np.float16) / 255.0
            target = (target.astype(np.float16) / 127.5) - 1.0
        else:
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, ref_img=source, src_img=source)


# 0403-add
class Teacher_Dataset(data.Dataset):
    # is_idt: None[test], False[previous], True[makeuodiffuse]
    def __init__(self, root=None, dir_test=None, phase='train', dim=(256, 256), is_idt=True):
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        self.random = None
        self.phase = phase
        self.dim = dim
        self.is_idt = is_idt
        self.dir_makeup = os.path.join(self.root, 'images')
        self.dir_nonmakeup = os.path.join(self.root, 'images')
        self.dir_seg = os.path.join(self.root, 'scgan_segs')
        # self.dir_seg = os.path.join(self.root, 'segs')
        self.makeup_names = []
        self.non_makeup_names = []
        self.dir_test = dir_test
        if self.phase == 'train':
            with open(os.path.join(self.root, 'makeup.txt'), 'r') as f:
                self.makeup_names = [name.strip() for name in f.readlines()]
            with open(os.path.join(self.root, 'non-makeup.txt'), 'r') as f:
                self.non_makeup_names = [name.strip() for name in f.readlines()]
        if self.phase == 'test':
            with open(self.dir_test, 'r') as f:
                for line in f.readlines():
                    non_makeup_name, make_upname = line.strip().split()
                    self.non_makeup_names.append(non_makeup_name)
                    self.makeup_names.append(make_upname)

    def pick(self):
        if self.random is not None:
            self.random = np.random.RandomState(np.random.seed())
            a_index = self.random.randint(0, len(self.makeup_names))
            another_index = self.random.randint(0, len(self.non_makeup_names))
        else:
            a_index = torch.randint(0, len(self.makeup_names), (1,)).item()
            another_index = torch.randint(0, len(self.non_makeup_names), (1,)).item()
        return [a_index, another_index]

    def __len__(self):
        if self.is_idt:
            return len(self.non_makeup_names) * 2 + len(self.makeup_names)
        elif not self.is_idt:
            return len(self.non_makeup_names) + len(self.makeup_names)
        if self.phase == 'train':
            return len(self.non_makeup_names)
        elif self.phase == 'test':
            return len(self.makeup_names)

    def __getitem__(self, index):
        if self.phase == 'test':
            makeup_name = self.makeup_names[index]
            nonmakeup_name = self.non_makeup_names[index]
        if self.phase == 'train':
            index = self.pick()
            makeup_name = self.makeup_names[index[0]]
            nonmakeup_name = self.non_makeup_names[index[1]]
        img_nonmakeup = cv2.imread(os.path.join(self.dir_nonmakeup, nonmakeup_name))
        img_makeup = cv2.imread(os.path.join(self.dir_makeup, makeup_name))
        img_nonmakeup_seg = cv2.imread(os.path.join(self.dir_seg, nonmakeup_name), 0)
        img_makeup_seg = cv2.imread(os.path.join(self.dir_seg, makeup_name), 0)

        img_nonmakeup = cv2.cvtColor(img_nonmakeup, cv2.COLOR_BGR2RGB)
        img_makeup = cv2.cvtColor(img_makeup, cv2.COLOR_BGR2RGB)
        prompt = 'makeup transfer'

        if self.dim:
            img_nonmakeup = cv2.resize(img_nonmakeup, self.dim, interpolation=cv2.INTER_AREA)
            img_makeup = cv2.resize(img_makeup, self.dim, interpolation=cv2.INTER_AREA)
            img_nonmakeup_seg = cv2.resize(img_nonmakeup_seg, self.dim, interpolation=cv2.INTER_AREA)
            img_makeup_seg = cv2.resize(img_makeup_seg, self.dim, interpolation=cv2.INTER_AREA)

        return_dict = {}
        return_dict['makeup_img'] = (img_makeup.astype(np.float32) / 127.5) - 1.0
        return_dict['nonmakeup_img'] = (img_nonmakeup.astype(np.float32) / 127.5) - 1.0
        return_dict['makeup_seg'] = img_makeup_seg
        return_dict['nonmakeup_seg'] = img_nonmakeup_seg
        return_dict['ref_img'] = img_makeup.astype(np.float32) / 255
        return_dict['src_img'] = img_nonmakeup.astype(np.float32) / 255
        return_dict['txt'] = prompt
        if not self.is_idt:
            # 0, 1, 2
            data_type = torch.randint(0, 3, (1,)).item()
            if data_type > 1:
                return_dict['makeup_img'] = (img_nonmakeup.astype(np.float32) / 127.5) - 1.0
                return_dict['nonmakeup_img'] = (img_makeup.astype(np.float32) / 127.5) - 1.0
                return_dict['nonmakeup_seg'] = img_makeup_seg
                return_dict['makeup_seg'] = img_nonmakeup_seg
                return_dict['src_img'] = img_makeup.astype(np.float32) / 255
                return_dict['ref_img'] = img_nonmakeup.astype(np.float32) / 255
        return return_dict


class Ele_PGT_Dataset(data.Dataset):
    def __init__(self, root=None, dim=(256, 256), img_size=256, keep_order=False):
        super(Ele_PGT_Dataset, self).__init__()
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        with open(os.path.join(self.root, 'makeup.txt'), 'r') as f:
            self.makeup_names = [name.strip() for name in f.readlines()]
        with open(os.path.join(self.root, 'non-makeup.txt'), 'r') as f:
            self.non_makeup_names = [name.strip() for name in f.readlines()]
        self.preprocessor = PreProcess()
        self.img_size = img_size
        self.dim = dim
        self.keep_order = keep_order

    def load_from_file(self, img_name):
        image = Image.open(os.path.join(self.root, 'images', img_name)).convert('RGB')
        mask = self.preprocessor.load_mask(os.path.join(self.root, 'segs', img_name))
        base_name = os.path.splitext(img_name)[0]
        lms = self.preprocessor.load_lms(os.path.join(self.root, 'lms', f'{base_name}.npy'))
        return self.preprocessor.process(image, mask, lms)

    def __len__(self):
        if self.keep_order:
            return max(len(self.makeup_names), len(self.non_makeup_names))
        else:
            return len(self.makeup_names) + len(self.non_makeup_names)

    def __getitem__(self, index):
        idx_s = torch.randint(0, len(self.non_makeup_names), (1,)).item()
        idx_r = torch.randint(0, len(self.makeup_names), (1,)).item()
        name_s = self.non_makeup_names[idx_s]
        name_r = self.makeup_names[idx_r]
        source = self.load_from_file(name_s)
        reference = self.load_from_file(name_r)
        img_nonmakeup_seg = cv2.imread(os.path.join(self.root, 'scgan_segs', name_s), 0)
        img_makeup_seg = cv2.imread(os.path.join(self.root, 'scgan_segs', name_r), 0)
        if self.dim:
            img_nonmakeup_seg = cv2.resize(img_nonmakeup_seg, self.dim, interpolation=cv2.INTER_AREA)
            img_makeup_seg = cv2.resize(img_makeup_seg, self.dim, interpolation=cv2.INTER_AREA)

        prompt = 'makeup transfer'
        return_dict = {}
        data_type = 0
        if not self.keep_order:
            data_type = torch.randint(0, 3, (1,)).item()
        # reference + source
        if data_type > 1:
            # source
            # list: img, mask_full, diff, landmarks
            return_dict['reference'] = source
            return_dict['makeup_img'] = source[0]                # [-1, 1]
            return_dict['makeup_seg'] = img_nonmakeup_seg        # [0, 14]
            return_dict['ref_img'] = (source[0] + 1) / 2         # [0, 1]
            # reference
            return_dict['source'] = reference
            return_dict['nonmakeup_img'] = reference[0]
            return_dict['nonmakeup_seg'] = img_makeup_seg
            return_dict['src_img'] = (reference[0] + 1) / 2
        # source + reference
        else:
            # source
            # list: img, mask_full, diff, landmarks
            return_dict['source'] = source
            return_dict['nonmakeup_img'] = source[0]             # [-1, 1]
            return_dict['nonmakeup_seg'] = img_nonmakeup_seg     # [0, 14]
            return_dict['src_img'] = (source[0] + 1) / 2         # [0, 1]
            # reference
            return_dict['reference'] = reference
            return_dict['makeup_img'] = reference[0]
            return_dict['makeup_seg'] = img_makeup_seg
            return_dict['ref_img'] = (reference[0] + 1) / 2
        return_dict['txt'] = prompt
        return return_dict


# 0411
class TestRandom_Dataset(data.Dataset):
    def __init__(self, root=None, dim=(256, 256), img_size=256):
        super(TestRandom_Dataset, self).__init__()
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        with open(os.path.join(self.root, 'makeup.txt'), 'r') as f:
            self.makeup_names = [name.strip() for name in f.readlines()]
        with open(os.path.join(self.root, 'non-makeup.txt'), 'r') as f:
            self.non_makeup_names = [name.strip() for name in f.readlines()]
        self.preprocessor = PreProcess()
        self.img_size = img_size
        self.dim = dim

    def load_from_file(self, img_name):
        image = Image.open(os.path.join(self.root, 'images', img_name)).convert('RGB')
        mask = self.preprocessor.load_mask(os.path.join(self.root, 'segs', img_name))
        base_name = os.path.splitext(img_name)[0]
        lms = self.preprocessor.load_lms(os.path.join(self.root, 'lms', f'{base_name}.npy'))
        return self.preprocessor.process(image, mask, lms)

    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def __getitem__(self, index):
        idx_s = torch.randint(0, len(self.non_makeup_names), (1,)).item()
        idx_r = torch.randint(0, len(self.makeup_names), (1,)).item()
        name_s = self.non_makeup_names[idx_s]
        name_r = self.makeup_names[idx_r]
        basename_s = os.path.basename(name_s).split('.')[0]
        basename_r = os.path.basename(name_r).split('.')[0]
        basename = '%s&%s' %(basename_s, basename_r)

        source = self.load_from_file(name_s)
        reference = self.load_from_file(name_r)
        img_nonmakeup_seg = cv2.imread(os.path.join(self.root, 'scgan_segs', name_s), 0)
        img_makeup_seg = cv2.imread(os.path.join(self.root, 'scgan_segs', name_r), 0)
        if self.dim:
            img_nonmakeup_seg = cv2.resize(img_nonmakeup_seg, self.dim, interpolation=cv2.INTER_AREA)
            img_makeup_seg = cv2.resize(img_makeup_seg, self.dim, interpolation=cv2.INTER_AREA)

        return_dict = {}
        prompt = 'makeup transfer'
        return_dict['source'] = source
        return_dict['nonmakeup_img'] = source[0]             # [-1, 1]
        return_dict['nonmakeup_seg'] = img_nonmakeup_seg     # [0, 14]
        return_dict['src_img'] = (source[0] + 1) / 2         # [0, 1]
        # reference
        return_dict['reference'] = reference
        return_dict['makeup_img'] = reference[0]
        return_dict['makeup_seg'] = img_makeup_seg
        return_dict['ref_img'] = (reference[0] + 1) / 2
        return_dict['txt'] = prompt
        return_dict['img_name'] = basename
        return return_dict


# 0411
class TestFixed_Dataset(data.Dataset):
    def __init__(self, root=None, dim=(256, 256), img_size=256, test_name=None):
        super(TestFixed_Dataset, self).__init__()
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        if test_name is None:
            test_name = 'test_0412.txt'

        with open(os.path.join(self.root, test_name), 'r') as f:
            self.makeup_names = [name.strip().split(' ')[1] for name in f.readlines()]
        with open(os.path.join(self.root, test_name), 'r') as f:
            self.non_makeup_names = [name.strip().split(' ')[0] for name in f.readlines()]
        self.preprocessor = PreProcess()
        self.img_size = img_size
        self.dim = dim

    def load_from_file(self, img_name):
        image = Image.open(os.path.join(self.root, 'images', img_name)).convert('RGB')
        mask = self.preprocessor.load_mask(os.path.join(self.root, 'segs', img_name))
        base_name = os.path.splitext(img_name)[0]
        lms = self.preprocessor.load_lms(os.path.join(self.root, 'lms', f'{base_name}.npy'))
        return self.preprocessor.process(image, mask, lms)

    def __len__(self):
        return len(self.makeup_names)

    def __getitem__(self, index):
        name_s = self.non_makeup_names[index]
        name_r = self.makeup_names[index]
        basename_s = os.path.basename(name_s).split('.')[0]
        basename_r = os.path.basename(name_r).split('.')[0]
        basename = '%s&%s' %(basename_s, basename_r)

        source = self.load_from_file(name_s)
        reference = self.load_from_file(name_r)
        img_nonmakeup_seg = cv2.imread(os.path.join(self.root, 'scgan_segs', name_s), 0)
        img_makeup_seg = cv2.imread(os.path.join(self.root, 'scgan_segs', name_r), 0)
        if self.dim:
            img_nonmakeup_seg = cv2.resize(img_nonmakeup_seg, self.dim, interpolation=cv2.INTER_AREA)
            img_makeup_seg = cv2.resize(img_makeup_seg, self.dim, interpolation=cv2.INTER_AREA)

        return_dict = {}
        prompt = 'makeup transfer'
        return_dict['source'] = source
        return_dict['nonmakeup_img'] = source[0]             # [-1, 1]
        return_dict['nonmakeup_seg'] = img_nonmakeup_seg     # [0, 14]
        return_dict['src_img'] = (source[0] + 1) / 2         # [0, 1]
        # reference
        return_dict['reference'] = reference
        return_dict['makeup_img'] = reference[0]
        return_dict['makeup_seg'] = img_makeup_seg
        return_dict['ref_img'] = (reference[0] + 1) / 2
        return_dict['txt'] = prompt
        return_dict['img_name'] = basename
        return return_dict



if __name__ == '__main__':
    dataset = TestFixed_Dataset()
    print(len(dataset))
    item = dataset[0]
    print(len(dataset), item['img_name'])