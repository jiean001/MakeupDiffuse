import os
import cv2
import torch
import numpy as np
import torch.utils.data as data
from scgan_models.SCGen import *
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Teacher_SCGAN(SCGen):
    def __init__(self, snapshot_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scgan_code_dir = os.environ['SCGAN']
        self.snapshot_path = os.path.join(scgan_code_dir, snapshot_path)
        self.load_checkpoint()
        self.PSEnc.phase = 'train'
        self.n_componets = n_componets

    def load_checkpoint(self):
        G_path = os.path.join(self.snapshot_path, 'G.pth')
        if os.path.exists(G_path):
            ckpt_state = torch.load(G_path)
            self.load_state_dict(ckpt_state)
            print('loaded trained generator {}..!'.format(G_path))
        self.cuda()

    def rebound_box(self, mask_A, mask_B, mask_A_face):
        mask_A = mask_A.unsqueeze(1)
        mask_B = mask_B.unsqueeze(1)
        mask_A_face = mask_A_face.unsqueeze(1)

        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[:, :, min(x_A_index) - 5:max(x_A_index) + 6, min(y_A_index) - 5:max(y_A_index) + 6] = \
            mask_A_face[:, :, min(x_A_index) - 5:max(x_A_index) + 6, min(y_A_index) - 5:max(y_A_index) + 6]
        mask_B_temp[:, :, min(x_B_index) - 5:max(x_B_index) + 6, min(y_B_index) - 5:max(y_B_index) + 6] = \
            mask_A_face[:, :, min(x_B_index) - 5:max(x_B_index) + 6, min(y_B_index) - 5:max(y_B_index) + 6]
        mask_A_temp = mask_A_temp.squeeze(1)
        mask_B_temp = mask_B_temp.squeeze(1)
        return mask_A_temp, mask_B_temp

    # mask_B: makeup_seg, mask_A: nonmakeup
    # makeup_img:B,C,H,W mask_B: B,H,W
    def set_input(self, makeup_img, nonmakeup_img, mask_B, mask_A):
        b, _, _, _ = makeup_img.shape
        self.makeup = makeup_img
        self.nonmakeup = nonmakeup_img
        self.makeup_seg = torch.zeros([self.n_componets, b, 256, 256], dtype=torch.float).to(mask_B.device)
        self.nonmakeup_seg = torch.zeros([self.n_componets, b, 256, 256], dtype=torch.float).to(mask_B.device)

        mask_A_lip = (mask_A == 9).float() + (mask_A == 13).float()
        mask_B_lip = (mask_B == 9).float() + (mask_B == 13).float()
        self.makeup_seg[0] = mask_B_lip
        self.nonmakeup_seg[0] = mask_A_lip

        mask_A_skin = (mask_A == 4).float() + (mask_A == 8).float() + (mask_A == 10).float()
        mask_B_skin = (mask_B == 4).float() + (mask_B == 8).float() + (mask_B == 10).float()
        self.makeup_seg[1] = mask_B_skin
        self.nonmakeup_seg[1] = mask_A_skin

        mask_A_eye_left = (mask_A == 6).float()
        mask_A_eye_right = (mask_A == 1).float()
        mask_B_eye_left = (mask_B == 6).float()
        mask_B_eye_right = (mask_B == 1).float()
        mask_A_face = (mask_A == 4).float() + (mask_A == 8).float()
        mask_B_face = (mask_B == 4).float() + (mask_B == 8).float()
        # avoid the es of ref are closed
        if not ((mask_B_eye_left > 0).any() and (mask_B_eye_right > 0).any()):
            return {}
        # todo understand why ...
        # mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
        self.makeup_seg[2] = mask_B_eye_left + mask_B_eye_right
        self.nonmakeup_seg[2] = mask_A_eye_left + mask_A_eye_right
        self.makeup_seg = rearrange(self.makeup_seg, 'c b h w -> b c h w')
        self.nonmakeup_seg = rearrange(self.nonmakeup_seg, 'c b h w -> b c h w')

    @torch.no_grad()
    def forward(self, makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg):
        self.set_input(makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg)
        makeup, nonmakeup = self.makeup, self.nonmakeup,
        makeup_seg, nonmakeup_seg = self.makeup_seg, self.nonmakeup_seg

        fid_x = self.FIEnc(nonmakeup)
        if self.ispartial or self.isinterpolation:
            exit()
        code = self.PSEnc(makeup, makeup_seg, makeup, makeup_seg, makeup, makeup_seg)
        result = self.fuse(fid_x, code, code)
        return result


class Teacher_Dataset(data.Dataset):
    def __init__(self, root=None, dir_test=None, phase='train', dim=(256, 256)):
        if root:
            self.root = root
        else:
            self.root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        self.random = None
        self.phase = phase
        self.dim = dim
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
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        a_index = self.random.randint(0, len(self.makeup_names))
        another_index = self.random.randint(0, len(self.non_makeup_names))
        return [a_index, another_index]

    def __len__(self):
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
        img_nonmakeup_seg = cv2.imread(os.path.join(self.dir_seg, makeup_name), 0)
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
        return return_dict


snapshot_path = "./checkpoints/"
dim = 64
style_dim = 192
n_downsample = 2
n_res = 3
mlp_dim = 256
n_componets = 3
input_dim = 3
phase = "test"
ispartial = False
isinterpolation = False
teacher = Teacher_SCGAN(snapshot_path=snapshot_path, dim=dim, style_dim=style_dim, n_downsample=n_downsample, n_res=n_res,
                        mlp_dim=mlp_dim, n_componets=n_componets, input_dim=input_dim, phase=phase, ispartial=ispartial, isinterpolation=isinterpolation)

dataset = Teacher_Dataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True, num_workers=4)

from einops import rearrange, repeat
import time

start_time = time.time()
# teacher = teacher.cuda()
device = 'cuda:1'
for i, data in enumerate(dataloader):
    print('load data:', time.time() - start_time)
    start_time = time.time()
    # makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg
    makeup_img = data['makeup_img'].cuda()
    nonmakeup_img = data['nonmakeup_img'].cuda()
    makeup_seg = data['makeup_seg'].cuda()
    nonmakeup_seg = data['nonmakeup_seg'].cuda()

    makeup_img = rearrange(makeup_img, 'b h w c -> b c h w')
    nonmakeup_img = rearrange(nonmakeup_img, 'b h w c -> b c h w')
    # makeup_seg = makeup_seg.unsqueeze(1)
    # nonmakeup_seg = nonmakeup_seg.unsqueeze(1)

    print('transfer to cuda:', time.time() - start_time)
    start_time = time.time()
    result = teacher(makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg)
    print('generate image:', time.time() - start_time)
    print(i)
    break
