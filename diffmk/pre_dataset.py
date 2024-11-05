import os
import numpy as np
from cldm.cldm import *
from cldm.utils import get_grid_image
from PIL import Image


class OnlyRec(ControlLDM):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        self.saved_root = '../../mkup_dataset/only_rec2/'

    def training_step(self, batch, batch_idx):
        self.check_input_and_output(batch)

    def get_origin_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = self.get_origin_input(batch, k)
        img_path = batch[self.cond_stage_key]
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z_encode = self.get_first_stage_encoding(encoder_posterior).detach()
        return x, z_encode, img_path

    def check_input_and_output(self, batch, N=4):
        x, z_encode, img_path = self.get_input(batch, self.first_stage_key, bs=N)
        reconstruct = self.decode_first_stage(z_encode)
        reconstruct = get_grid_image(reconstruct)
        img_path_split = img_path[0].split('/')

        saved_dir = os.path.join(self.saved_root, img_path_split[-2])
        os.makedirs(os.path.join(saved_dir, 'ori'), exist_ok=True)
        os.makedirs(os.path.join(saved_dir, 'rec'), exist_ok=True)

        Image.fromarray(reconstruct).save(os.path.join(saved_dir, 'ori', img_path_split[-1]))
        Image.fromarray(reconstruct).save(os.path.join(saved_dir, 'rec', img_path_split[-1]))
        return None


class InvRec(ControlLDM):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        self.saved_root = '../../mkup_dataset/InvRecColorGray80-40/'
        self.t0 = 80
        self.ddim_sampler = None
        self.diffusion_steps = 40
        self.t_encode = 40

    def update_schedule(self):
        self.register_schedule(given_betas=None, beta_schedule='linear', timesteps=self.t0,
                               linear_start=self.linear_start, linear_end=self.linear_end, cosine_s=8e-3)

    def get_ddim_sampler(self):
        if self.ddim_sampler:
            return self.ddim_sampler
        else:
            self.update_schedule()
            self.ddim_sampler = DDIMSampler(self)
            self.ddim_sampler.make_schedule(ddim_num_steps=40)
            return self.ddim_sampler

    def training_step(self, batch, batch_idx):
        self.check_input_and_output(batch)

    def get_t_encode(self):
        seq_inv = np.linspace(0, 1, self.diffusion_steps) * self.num_timesteps
        seq_inv = [int(s) for s in list(seq_inv)]
        # seq_inv_next = [-1] + list(seq_inv[:-1])
        return seq_inv

    def check_input_and_output(self, batch, N=4):
        # dict(c_crossattn=[c], c_concat=[control])
        z_encode, c = self.get_input(batch, self.first_stage_key)
        c['c_concat'] = None
        img_path = batch['path']
        # x0, c, t_enc
        # x_latent, cond, t_start
        # z_encode_inters
        ddim_sampler = self.get_ddim_sampler()
        z_encode_inv, _ = ddim_sampler.encode(x0=z_encode, c=c, t_enc=self.t_encode)
        z_encode_gen = ddim_sampler.decode(x_latent=z_encode_inv, cond=c, t_start=self.diffusion_steps)
        # DDIM Decode
        origin = self.decode_first_stage(z_encode)
        origin = get_grid_image(origin)
        reconstruct = self.decode_first_stage(z_encode_gen)
        reconstruct = get_grid_image(reconstruct)
        img_path_split = img_path[0].split('/')

        saved_dir = os.path.join(self.saved_root, img_path_split[-2])
        os.makedirs(os.path.join(saved_dir, 'ori'), exist_ok=True)
        os.makedirs(os.path.join(saved_dir, 'rec'), exist_ok=True)
        os.makedirs(os.path.join(saved_dir, 'inv'), exist_ok=True)
        os.makedirs(os.path.join(saved_dir, 'gen'), exist_ok=True)

        Image.fromarray(origin).save(os.path.join(saved_dir, 'ori', img_path_split[-1]))
        Image.fromarray(reconstruct).save(os.path.join(saved_dir, 'rec', img_path_split[-1]))
        torch.save(z_encode_inv.detach().cpu(), os.path.join(saved_dir, 'inv', '%s.pth' % (img_path_split[-1])))
        torch.save(z_encode_gen.detach().cpu(), os.path.join(saved_dir, 'gen', '%s.pth' % (img_path_split[-1])))
        return None

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None,
                              only_mid_control=self.only_mid_control)
        return eps