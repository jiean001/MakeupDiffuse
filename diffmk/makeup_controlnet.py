import os
import torch
import numpy as np
from cldm.cldm import *
from diffmk.cddim import MKDDIMSampler
from diffmk.histogram_matching import *
from diffmk.utils import get_grid_image
from PIL import Image


class BaseModel(ControlLDM):
    def __init__(self, control_src_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_src_key = control_src_key
        # self.control_src_model = instantiate_from_config(control_src_stage_config)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # x: [1, 4, 64, 64]
        # c: [1, 77, 768]
        # Encoder(x)
        x, c = super().get_input(batch, k, bs, *args, **kwargs)
        control_src = batch[self.control_src_key]
        if bs is not None:
            control_src = control_src[:bs]
        control_src = control_src.to(self.device)
        control_src = einops.rearrange(control_src, 'b h w c -> b c h w')
        control_src = control_src.to(memory_format=torch.contiguous_format).float()
        control_ref = c['c_concat'][0]
        control = torch.cat((control_src, control_ref), 1)
        c['c_concat'] = [control]
        return x, c

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        # c["c_concat"][0] B 6 H W
        # ref
        # src
        c_cat, c_src, c = c["c_concat"][0][:N,:3], c["c_concat"][0][:N,3:], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control_ref"] = c_cat * 2.0 - 1.0
        log["control_src"] = c_src * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((256, 256), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            c_cat = torch.cat([c_cat, c_src], 1)
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            c_cat = torch.cat([c_cat, c_src], 1)
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log


class MakeupDoubleControlModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_origin_img_input(self, batch, k, need_rearrange=True):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(self.device)
        if need_rearrange:
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_origin_img_input(self, batch, k, need_rearrange=True):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(self.device)
        if need_rearrange:
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_cond_txt_coding(self, batch, force_c_encode=False):
        xc = batch[self.cond_stage_key]
        if not self.cond_stage_trainable or force_c_encode:
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                c = self.get_learned_conditioning(xc.to(self.device))
        else:
            c = xc
        return c

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = self.get_origin_img_input(batch, self.first_stage_key, need_rearrange=False)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        # prompt
        txt_ctl = self.get_cond_txt_coding(batch)
        if bs is not None:
            txt_ctl = txt_ctl[:bs]

        # source
        control_src = batch[self.control_src_key]
        if bs is not None:
            control_src = control_src[:bs]
        control_src = control_src.to(self.device)
        control_src = einops.rearrange(control_src, 'b h w c -> b c h w')
        control_src = control_src.to(memory_format=torch.contiguous_format).float()

        # reference
        control_ref = batch[self.control_key]
        if bs is not None:
            control_ref = control_ref[:bs]
        control_ref = control_ref.to(self.device)
        control_ref = einops.rearrange(control_ref, 'b h w c -> b c h w')
        control_ref = control_ref.to(memory_format=torch.contiguous_format).float()

        return z, dict(c_crossattn=[txt_ctl], c_concat=[torch.cat([control_src, control_ref], 1)])
