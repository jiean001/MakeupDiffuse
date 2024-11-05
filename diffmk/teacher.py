import argparse
from scgan_models.SCGen import *
from einops import rearrange
from ele_models.loss import AnnealingComposePGT
from ele_training.inference import Inference
from ele_training.config import get_config


class Teacher_SCGAN(SCGen):
    def __init__(self, snapshot_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scgan_code_dir = os.environ['SCGAN']
        self.snapshot_path = os.path.join(scgan_code_dir, snapshot_path)
        self.load_checkpoint()
        self.PSEnc.phase = 'train'

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
        # mask_A_face = (mask_A == 4).float() + (mask_A == 8).float()
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


class Ele_PGT_Teacher(AnnealingComposePGT):
    def __init__(self):
        EYE_MARGIN = 12
        LIP_MARGIN = 4
        SKIN_ALPHA = 0.3
        SKIN_ALPHA_MILESTONES = (0, 12, 24, 50)
        SKIN_ALPHA_VALUES = (0.2, 0.4, 0.3, 0.2)
        EYE_ALPHA = 0.8
        EYE_ALPHA_MILESTONES = (0, 12, 24, 50)
        EYE_ALPHA_VALUES = (0.6, 0.8, 0.6, 0.4)
        LIP_ALPHA = 0.1
        LIP_ALPHA_MILESTONES = (0, 12, 24, 50)
        LIP_ALPHA_VALUES = (0.05, 0.2, 0.1, 0.0)
        self.margins = {'eye': EYE_MARGIN, 'lip': LIP_MARGIN}
        super().__init__(self.margins, SKIN_ALPHA_MILESTONES, SKIN_ALPHA_VALUES, EYE_ALPHA_MILESTONES,
                         EYE_ALPHA_VALUES, LIP_ALPHA_MILESTONES, LIP_ALPHA_VALUES)
        self.eval()


class EleGANt_Teacher(Inference):
    def __init__(self, model_path="G.pth"):
        config = get_config()
        parser = argparse.ArgumentParser("argument for training")
        args = parser.parse_args(args=[])
        code_dir = os.environ['ELEGANT']
        args.name = 'our_test_0320-01-author'
        args.save_path = os.path.join(code_dir, 'result')
        # author: sow_pyramid_a5_e3d2_remapped.pth
        # ours 0318: A100_0318_epoch50_G.pth
        args.load_path = os.path.join(code_dir, 'ckpts/sow_pyramid_a5_e3d2_remapped.pth')
        data_root = os.path.join(os.environ['DATAROOT'], 'MT-Dataset')
        args.source_dir = os.path.join(data_root, 'images')
        args.reference_dir = os.path.join(data_root, 'images')
        args.device = 'cpu'
        args.save_folder = os.path.join(args.save_path, args.name)

        super(EleGANt_Teacher, self).__init__(config, args, args.load_path)
        self.solver.G.cuda()

    @torch.no_grad()
    def transfer(self, image_s, image_r, mask_s_full, mask_r_full, diff_s, diff_r, lms_s, lms_r):
        mask_s = torch.cat((mask_s_full[:, 0:1], mask_s_full[:, 1:].sum(dim=1, keepdim=True)), dim=1)
        mask_r = torch.cat((mask_r_full[:, 0:1], mask_r_full[:, 1:].sum(dim=1, keepdim=True)), dim=1)
        fake_A = self.solver.generate(image_s, image_r, mask_s, mask_r, diff_s, diff_r, lms_s, lms_r)
        return fake_A


class Source_Teacher():
    def __init__(self):
        pass

    def keep_source(self, image_s):
        return image_s


def test_elegant_teacher():
    from diffdata.datasets import Ele_PGT_Dataset
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    dataset = Ele_PGT_Dataset()
    dataloader = DataLoader(dataset, num_workers=2, batch_size=6, shuffle=True)
    print(len(dataset), len(dataset[0]), dataset[0].keys())
    teacher = EleGANt_Teacher()

    def set_input(source, reference, device='cuda'):
        image_s, image_r = source[0], reference[0]  # (b, c, h, w)
        mask_s_full, mask_r_full = source[1], reference[1]  # (b, c', h, w)
        diff_s, diff_r = source[2], reference[2]  # (b, 136, h, w)
        lms_s, lms_r = source[3], reference[3]  # (b, K, 2)

        image_s = image_s.to(device)
        image_r = image_r.to(device)
        mask_s_full = mask_s_full.to(device)
        mask_r_full = mask_r_full.to(device)
        diff_s = diff_s.to(device)
        diff_r = diff_r.to(device)
        lms_s = lms_s.to(device)
        lms_r = lms_r.to(device)

        image_s = image_s.to(memory_format=torch.contiguous_format).float()
        image_r = image_r.to(memory_format=torch.contiguous_format).float()
        mask_s_full = mask_s_full.to(memory_format=torch.contiguous_format).float()
        mask_r_full = mask_r_full.to(memory_format=torch.contiguous_format).float()
        diff_s = diff_s.to(memory_format=torch.contiguous_format).float()
        diff_r = diff_r.to(memory_format=torch.contiguous_format).float()
        lms_s = lms_s.to(memory_format=torch.contiguous_format).float()
        lms_r = lms_r.to(memory_format=torch.contiguous_format).float()
        return image_s, image_r, mask_s_full, mask_r_full, diff_s, diff_r, lms_s, lms_r

    for data in dataloader:
        source = data['source']
        reference = data['reference']
        image_s, image_r, mask_s_full, mask_r_full, diff_s, diff_r, lms_s, lms_r = set_input(source, reference)
        image = teacher.transfer(image_s, image_r, mask_s_full, mask_r_full, diff_s, diff_r, lms_s, lms_r)
        image =image.clamp(-1, 1)
        save_image((image+1)/2, 'test_teacher.png')
        break

if __name__ == '__main__':
    # test_elegant_teacher()
    pass
