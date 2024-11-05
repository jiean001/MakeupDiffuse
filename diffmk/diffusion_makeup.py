from typing import Optional, Any
import numpy as np
import torch
import os
import torchvision
from pytorch_lightning.utilities.types import STEP_OUTPUT
from PIL import Image
from diffmk.makeup_teacher import *
from diffmk.makeup_diffuse import PGTBaseModel
from ele_models.modules.pseudo_gt import expand_area
from pytorch_lightning.utilities.distributed import rank_zero_only


class BaseDoubleControlModel(PGTBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_elegant_target(self, batch):
        target = self.teacher_model.transfer(self.image_s, self.image_r, self.mask_s_full, self.mask_r_full,
                                             self.diff_s, self.diff_r, self.lms_s, self.lms_r)
        return target

    def get_keep_target(self):
        target = self.teacher_model.keep_source(self.image_s)
        return target

    def set_pgt_input(self, source, reference, device):
        image_s, image_r = source[0], reference[0]  # (b, c, h, w)
        mask_s_full, mask_r_full = source[1], reference[1]  # (b, c', h, w)
        diff_s, diff_r = source[2], reference[2]
        lms_s, lms_r = source[3], reference[3]  # (b, K, 2)

        image_s = image_s.to(device)
        image_r = image_r.to(device)
        mask_s_full = mask_s_full.to(device)
        mask_r_full = mask_r_full.to(device)
        diff_s = diff_s.to(device)
        diff_r = diff_r.to(device)
        lms_s = lms_s.to(device)
        lms_r = lms_r.to(device)

        self.image_s = image_s.to(memory_format=torch.contiguous_format).float()
        self.image_r = image_r.to(memory_format=torch.contiguous_format).float()
        self.mask_s_full = mask_s_full.to(memory_format=torch.contiguous_format).float()
        self.mask_r_full = mask_r_full.to(memory_format=torch.contiguous_format).float()
        self.diff_s = diff_s.to(memory_format=torch.contiguous_format).float()
        self.diff_r = diff_r.to(memory_format=torch.contiguous_format).float()
        self.lms_s = lms_s.to(memory_format=torch.contiguous_format).float()
        self.lms_r = lms_r.to(memory_format=torch.contiguous_format).float()

    def get_target(self, batch, is_fixbkgrd=False):
        self.makeup_pgt = None
        makeup_img = self.get_origin_img_input(batch, self.makeup_img_key)
        nonmakeup_img = self.get_origin_img_input(batch, self.nonmakeup_img_key)
        makeup_seg = self.get_seg_img_input(batch, self.makeup_seg_key)
        nonmakeup_seg = self.get_seg_img_input(batch, self.nonmakeup_seg_key)
        if self.teacher_type == 'SCGAN':
            target = self.teacher_model(makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg)
            target = target.clamp(-1, 1)
        elif self.teacher_type == 'ELEGANT':
            source = batch[self.source_key]
            reference = batch[self.reference_key]
            self.set_pgt_input(source, reference, self.device)
            target = self.get_elegant_target(batch)
            target = target.clamp(-1, 1)
            if self.w_bkgrd > 0 or self.w_makeup > 0:
                self.makeup_pgt = self.pgt_model(self.image_s, self.image_r, self.mask_s_full, self.mask_r_full,
                                                 self.lms_s, self.lms_r)
        elif self.teacher_type == 'ELEGANT_PGT':
            source = batch[self.source_key]
            reference = batch[self.reference_key]
            self.set_pgt_input(source, reference, self.device)
            target = self.pgt_model(self.image_s, self.image_r, self.mask_s_full, self.mask_r_full, self.lms_s,
                                    self.lms_r)
            target = target.clamp(-1, 1)
            if self.w_bkgrd > 0 or self.w_makeup > 0:
                self.makeup_pgt = target
        elif self.teacher_type == 'KEEP':
            source = batch[self.source_key]
            reference = batch[self.reference_key]
            self.set_pgt_input(source, reference, self.device)
            target = self.get_keep_target()
            target = target.clamp(-1, 1)
            if self.w_bkgrd > 0 or self.w_makeup > 0:
                self.makeup_pgt = self.pgt_model(self.image_s, self.image_r, self.mask_s_full, self.mask_r_full,
                                                 self.lms_s, self.lms_r)
        if (self.w_bkgrd > 0 or self.w_makeup > 0) and self.makeup_pgt is None:
            source = batch[self.source_key]
            reference = batch[self.reference_key]
            self.set_pgt_input(source, reference, self.device)
            self.makeup_pgt = self.pgt_model(self.image_s, self.image_r, self.mask_s_full, self.mask_r_full, self.lms_s,
                                    self.lms_r)
            self.makeup_pgt = self.makeup_pgt.clamp(-1, 1)
        return target, makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg


class BaseSingleControlModel(BaseDoubleControlModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        src_img = self.get_origin_img_input(batch, self.src_img_key)
        ref_img = self.get_origin_img_input(batch, self.ref_img_key)
        pgt_sr, makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg = self.get_target(batch)
        txt_ctl = self.get_cond_txt_coding(batch)
        c = {}
        c['c_crossattn'] = [txt_ctl]
        c['pgt_sr'] = pgt_sr
        c['makeup_img'] = makeup_img
        c['nonmakeup_img'] = nonmakeup_img
        c['makeup_seg'] = makeup_seg
        c['nonmakeup_seg'] = nonmakeup_seg
        c['src_img'] = src_img
        c['ref_img'] = ref_img
        # c['c_concat'] = [torch.cat((src_img, ref_img), 1)]
        c['c_concat'] = [ref_img]
        return None, c

    def p_loss_diffuse(self, tmin, gt, cond, src, ref):
        z = self.get_z(gt)
        t = torch.randint(tmin, self.num_timesteps, (z.shape[0],), device=self.device).long()
        cond['c_concat'] = [ref]
        loss, fake_sr_z = self.p_loss_diffuse_base(z, cond, t)
        return loss, fake_sr_z

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        # todo
        _, c = self.get_input(batch, self.first_stage_key, bs=N)
        pgt_sr = c['pgt_sr']
        z = self.get_z(pgt_sr)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        log["reconstruction"] = self.decode_first_stage(z)
        log["control_ref"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((256, 256), batch[self.cond_stage_key], size=22)
        log["ground_truth"] = pgt_sr[:N]

        z_start = z[:N]
        t = torch.randint(self.t_min, self.num_timesteps, (z_start.shape[0],), device=self.device).long()
        noise = default(None, lambda: torch.randn_like(z_start))
        x_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        _, x_recon = self.apply_model(x_noisy, t, {"c_concat": [c_cat], "c_crossattn": [c]}, return_all=True)
        log["sample_ddmp"] = self.decode_first_stage(x_recon)

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
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
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


class TestSingleControlModel(BaseSingleControlModel):
    def __init__(self, saved_dir, model_name, img_name_key, unconditional_guidance_scale=9, ddim_steps=50, ddim_eta=0.0,
                 sample=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.ddim_steps = ddim_steps
        self.sample = sample
        self.ddim_eta = ddim_eta
        self.model_name = model_name
        self.saved_dir = saved_dir
        self.img_name_key = img_name_key
        self.clamp = True
        self.rescale = True
        self.test_pairs = []

    def on_test_epoch_start(self) -> None:
        self.eval()
        self.test_pairs = []

    def on_test_batch_end(self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # num-num nonmakeup makeup
        with open('test_0412_pairs.txt', 'w') as f:
            for test_pair in self.test_pairs:
                f.write('%s %s %s\n' %(test_pair[0], test_pair[1], test_pair[2]))

    def test_step(self, batch, batch_idx):
        images = self.log_results(batch, batch_idx)
        for k in images:
            N = images[k].shape[0]
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)
        self.save_local(images, batch_idx)


    @rank_zero_only
    def save_local(self, images, batch_idx):
        root = os.path.join(self.saved_dir, self.model_name)
        nrow = len(images)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=nrow)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_{:04}.png".format(k, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @torch.no_grad()
    def log_results(self, batch, batch_idx):
        use_ddim = self.ddim_steps is not None

        log = dict()
        _, c = self.get_input(batch, self.first_stage_key)
        pgt_sr = c['pgt_sr']
        z = self.get_z(pgt_sr)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["reconstruction"] = self.decode_first_stage(z)
        src, ref = torch.chunk(c_cat, 2, dim=1)
        log["control_src"] = src * 2.0 - 1.0
        log["control_ref"] = ref * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((256, 256), batch[self.img_name_key], size=16)
        log["ground_truth"] = pgt_sr

        img_name_pairs = batch[self.img_name_key]
        for i in range(len(img_name_pairs)):
            test_pair = ['%04d-%d' %(batch_idx, i + 1)]
            test_pair.append('non-makeup/%s.png' % (img_name_pairs[i].split('&')[0]))
            test_pair.append('makeup/%s.png' %(img_name_pairs[i].split('&')[1]))
            self.test_pairs.append(test_pair)

        z_start = z
        t = torch.randint(self.t_min, self.num_timesteps, (z_start.shape[0],), device=self.device).long()
        noise = default(None, lambda: torch.randn_like(z_start))
        x_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        _, x_recon = self.apply_model(x_noisy, t, {"c_concat": [c_cat], "c_crossattn": [c]}, return_all=True)
        log["sample_ddmp"] = self.decode_first_stage(x_recon)
        b = pgt_sr.shape[0]
        if self.sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=b, ddim=use_ddim,
                                                     ddim_steps=self.ddim_steps, eta=self.ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

        if self.unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(b)  # c
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=b, ddim=use_ddim,
                                             ddim_steps=self.ddim_steps, eta=self.ddim_eta,
                                             unconditional_guidance_scale=self.unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{self.unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log


class TestDoubleControlModel(BaseDoubleControlModel):
    def __init__(self, saved_dir, model_name, img_name_key, unconditional_guidance_scale=9, ddim_steps=50, ddim_eta=0.0,
                 sample=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.ddim_steps = ddim_steps
        self.sample = sample
        self.ddim_eta = ddim_eta
        self.model_name = model_name
        self.saved_dir = saved_dir
        self.img_name_key = img_name_key
        self.clamp = True
        self.rescale = True
        self.test_pairs = []

    def on_test_epoch_start(self) -> None:
        self.eval()
        self.test_pairs = []

    def on_test_batch_end(self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # num-num nonmakeup makeup
        with open('test_0412_pairs.txt', 'w') as f:
            for test_pair in self.test_pairs:
                f.write('%s %s %s\n' %(test_pair[0], test_pair[1], test_pair[2]))

    def test_step(self, batch, batch_idx):
        images = self.log_results(batch, batch_idx)
        for k in images:
            N = images[k].shape[0]
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)
        self.save_local(images, batch_idx)


    @rank_zero_only
    def save_local(self, images, batch_idx):
        root = os.path.join(self.saved_dir, self.model_name)
        nrow = len(images)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=nrow)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_{:04}.png".format(k, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @torch.no_grad()
    def log_results(self, batch, batch_idx):
        use_ddim = self.ddim_steps is not None

        log = dict()
        _, c = self.get_input(batch, self.first_stage_key)
        pgt_sr = c['pgt_sr']
        z = self.get_z(pgt_sr)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["reconstruction"] = self.decode_first_stage(z)
        src, ref = torch.chunk(c_cat, 2, dim=1)
        log["control_src"] = src * 2.0 - 1.0
        log["control_ref"] = ref * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((256, 256), batch[self.img_name_key], size=16)
        log["ground_truth"] = pgt_sr

        img_name_pairs = batch[self.img_name_key]
        for i in range(len(img_name_pairs)):
            test_pair = ['%04d-%d' %(batch_idx, i + 1)]
            test_pair.append('non-makeup/%s.png' % (img_name_pairs[i].split('&')[0]))
            test_pair.append('makeup/%s.png' %(img_name_pairs[i].split('&')[1]))
            self.test_pairs.append(test_pair)

        z_start = z
        t = torch.randint(self.t_min, self.num_timesteps, (z_start.shape[0],), device=self.device).long()
        noise = default(None, lambda: torch.randn_like(z_start))
        x_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        _, x_recon = self.apply_model(x_noisy, t, {"c_concat": [c_cat], "c_crossattn": [c]}, return_all=True)
        log["sample_ddmp"] = self.decode_first_stage(x_recon)
        b = pgt_sr.shape[0]
        if self.sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=b, ddim=use_ddim,
                                                     ddim_steps=self.ddim_steps, eta=self.ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

        if self.unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(b)  # c
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=b, ddim=use_ddim,
                                             ddim_steps=self.ddim_steps, eta=self.ddim_eta,
                                             unconditional_guidance_scale=self.unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{self.unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log


class NoControlModel(BaseDoubleControlModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_all=False, *args, **kwargs):
        cond['c_concat'] = None
        if not return_all:
            eps = super(BaseModel, self).apply_model(x_noisy, t, cond, *args, **kwargs)
            return eps
        else:
            assert isinstance(cond, dict)
            diffusion_model = self.model.diffusion_model
            cond_txt = torch.cat(cond['c_crossattn'], 1)
            if cond['c_concat'] is None:
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None,
                                      only_mid_control=self.only_mid_control)
            else:
                control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t,
                                             context=cond_txt)
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control,
                                      only_mid_control=self.only_mid_control)
            x_recon = self.predict_start_from_noise(x_t=x_noisy, t=t, noise=eps)
            return eps, x_recon


class SingleControlModel(BaseDoubleControlModel):
    def __init__(self, single_control_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_control_key = single_control_key

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        src_img = self.get_origin_img_input(batch, self.src_img_key)
        ref_img = self.get_origin_img_input(batch, self.ref_img_key)
        pgt_sr, makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg = self.get_target(batch)
        txt_ctl = self.get_cond_txt_coding(batch)
        c = {}
        c['c_crossattn'] = [txt_ctl]
        c['pgt_sr'] = pgt_sr
        c['makeup_img'] = makeup_img
        c['nonmakeup_img'] = nonmakeup_img
        c['makeup_seg'] = makeup_seg
        c['nonmakeup_seg'] = nonmakeup_seg
        c['src_img'] = src_img
        c['ref_img'] = ref_img
        # c['c_concat'] = [torch.cat((src_img, ref_img), 1)]
        if self.single_control_key == self.ref_img_key:
            c['c_concat'] = [ref_img]
        elif self.single_control_key == self.src_img_key:
            c['c_concat'] = [src_img]
        else:
            assert 1 == 2
            # c['c_concat'] = [torch.cat((src_img, ref_img), 1)]
        return None, c

    def p_loss_diffuse(self, tmin, gt, cond, src, ref):
        z = self.get_z(gt)
        t = torch.randint(tmin, self.num_timesteps, (z.shape[0],), device=self.device).long()
        if self.single_control_key == self.ref_img_key:
            cond['c_concat'] = [ref]
        elif self.single_control_key == self.src_img_key:
            cond['c_concat'] = [src]
        else:
            assert 1 == 2
            # c['c_concat'] = [torch.cat((src_img, ref_img), 1)]
        loss, fake_sr_z = self.p_loss_diffuse_base(z, cond, t)
        return loss, fake_sr_z

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        # todo
        _, c = self.get_input(batch, self.first_stage_key, bs=N)
        pgt_sr = c['pgt_sr']
        src = c['src_img']
        ref = c['ref_img']
        z = self.get_z(pgt_sr)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        log["reconstruction"] = self.decode_first_stage(z)
        src = src[:N]
        ref = ref[:N]
        log["control_src"] = src * 2.0 - 1.0
        log["control_ref"] = ref * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((256, 256), batch[self.cond_stage_key], size=22)
        log["ground_truth"] = pgt_sr[:N]

        z_start = z[:N]
        t = torch.randint(self.t_min, self.num_timesteps, (z_start.shape[0],), device=self.device).long()
        noise = default(None, lambda: torch.randn_like(z_start))
        x_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        _, x_recon = self.apply_model(x_noisy, t, {"c_concat": [c_cat], "c_crossattn": [c]}, return_all=True)
        log["sample_ddmp"] = self.decode_first_stage(x_recon)

        if unconditional_guidance_scale > 1.0:
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


class TestSingleControlModelTVCJ(SingleControlModel):
    def __init__(self, saved_dir, model_name, img_name_key, unconditional_guidance_scale=9, ddim_steps=50, ddim_eta=0.0,
                 sample=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.ddim_steps = ddim_steps
        self.sample = sample
        self.ddim_eta = ddim_eta
        self.model_name = model_name
        self.saved_dir = saved_dir
        self.img_name_key = img_name_key
        self.clamp = True
        self.rescale = True
        self.test_pairs = []

    def on_test_epoch_start(self) -> None:
        self.eval()
        self.test_pairs = []

    def on_test_batch_end(self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # num-num nonmakeup makeup
        with open('test_0412_pairs.txt', 'w') as f:
            for test_pair in self.test_pairs:
                f.write('%s %s %s\n' %(test_pair[0], test_pair[1], test_pair[2]))

    def test_step(self, batch, batch_idx):
        images = self.log_results(batch, batch_idx)
        for k in images:
            N = images[k].shape[0]
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)
        self.save_local(images, batch_idx)


    @rank_zero_only
    def save_local(self, images, batch_idx):
        root = os.path.join(self.saved_dir, self.model_name)
        nrow = len(images)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=nrow)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_{:04}.png".format(k, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @torch.no_grad()
    def log_results(self, batch, batch_idx):
        use_ddim = self.ddim_steps is not None

        log = dict()
        _, c = self.get_input(batch, self.first_stage_key)
        pgt_sr = c['pgt_sr']
        src = c['src_img']
        ref = c['ref_img']
        z = self.get_z(pgt_sr)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["reconstruction"] = self.decode_first_stage(z)
        # src, ref = torch.chunk(c_cat, 2, dim=1)
        log["control_src"] = src * 2.0 - 1.0
        log["control_ref"] = ref * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((256, 256), batch[self.img_name_key], size=16)
        log["ground_truth"] = pgt_sr

        img_name_pairs = batch[self.img_name_key]
        for i in range(len(img_name_pairs)):
            test_pair = ['%04d-%d' %(batch_idx, i + 1)]
            test_pair.append('non-makeup/%s.png' % (img_name_pairs[i].split('&')[0]))
            test_pair.append('makeup/%s.png' %(img_name_pairs[i].split('&')[1]))
            self.test_pairs.append(test_pair)

        z_start = z
        t = torch.randint(self.t_min, self.num_timesteps, (z_start.shape[0],), device=self.device).long()
        noise = default(None, lambda: torch.randn_like(z_start))
        x_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        _, x_recon = self.apply_model(x_noisy, t, {"c_concat": [c_cat], "c_crossattn": [c]}, return_all=True)
        log["sample_ddmp"] = self.decode_first_stage(x_recon)
        b = pgt_sr.shape[0]
        if self.sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=b, ddim=use_ddim,
                                                     ddim_steps=self.ddim_steps, eta=self.ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

        if self.unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(b)  # c
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=b, ddim=use_ddim,
                                             ddim_steps=self.ddim_steps, eta=self.ddim_eta,
                                             unconditional_guidance_scale=self.unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{self.unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log


class TestNoControlModel(NoControlModel):
    def __init__(self, saved_dir, model_name, img_name_key, unconditional_guidance_scale=9, ddim_steps=50, ddim_eta=0.0,
                 sample=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.ddim_steps = ddim_steps
        self.sample = sample
        self.ddim_eta = ddim_eta
        self.model_name = model_name
        self.saved_dir = saved_dir
        self.img_name_key = img_name_key
        self.clamp = True
        self.rescale = True
        self.test_pairs = []

    def on_test_epoch_start(self) -> None:
        self.eval()
        self.test_pairs = []

    def on_test_batch_end(self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # num-num nonmakeup makeup
        with open('test_0412_pairs.txt', 'w') as f:
            for test_pair in self.test_pairs:
                f.write('%s %s %s\n' %(test_pair[0], test_pair[1], test_pair[2]))

    def test_step(self, batch, batch_idx):
        images = self.log_results(batch, batch_idx)
        for k in images:
            N = images[k].shape[0]
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)
        self.save_local(images, batch_idx)


    @rank_zero_only
    def save_local(self, images, batch_idx):
        root = os.path.join(self.saved_dir, self.model_name)
        nrow = len(images)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=nrow)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_{:04}.png".format(k, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @torch.no_grad()
    def log_results(self, batch, batch_idx):
        use_ddim = self.ddim_steps is not None

        log = dict()
        _, c = self.get_input(batch, self.first_stage_key)
        pgt_sr = c['pgt_sr']
        z = self.get_z(pgt_sr)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["reconstruction"] = self.decode_first_stage(z)
        src, ref = torch.chunk(c_cat, 2, dim=1)
        log["control_src"] = src * 2.0 - 1.0
        log["control_ref"] = ref * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((256, 256), batch[self.img_name_key], size=16)
        log["ground_truth"] = pgt_sr

        img_name_pairs = batch[self.img_name_key]
        for i in range(len(img_name_pairs)):
            test_pair = ['%04d-%d' %(batch_idx, i + 1)]
            test_pair.append('non-makeup/%s.png' % (img_name_pairs[i].split('&')[0]))
            test_pair.append('makeup/%s.png' %(img_name_pairs[i].split('&')[1]))
            self.test_pairs.append(test_pair)

        z_start = z
        t = torch.randint(self.t_min, self.num_timesteps, (z_start.shape[0],), device=self.device).long()
        noise = default(None, lambda: torch.randn_like(z_start))
        x_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        _, x_recon = self.apply_model(x_noisy, t, {"c_concat": [c_cat], "c_crossattn": [c]}, return_all=True)
        log["sample_ddmp"] = self.decode_first_stage(x_recon)
        b = pgt_sr.shape[0]
        if self.sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=b, ddim=use_ddim,
                                                     ddim_steps=self.ddim_steps, eta=self.ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

        if self.unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(b)  # c
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=b, ddim=use_ddim,
                                             ddim_steps=self.ddim_steps, eta=self.ddim_eta,
                                             unconditional_guidance_scale=self.unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{self.unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log