from cldm.cldm import *
from ldm.util import default


class BaseModel(ControlLDM):
    def __init__(self, teacher_config, src_key, makeup_img_key, nonmakeup_img_key, makeup_seg_key, nonmakeup_seg_key,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = instantiate_from_config(teacher_config)
        self.src_img_key = src_key
        self.makeup_img_key = makeup_img_key
        self.nonmakeup_img_key = nonmakeup_img_key
        self.makeup_seg_key = makeup_seg_key
        self.nonmakeup_seg_key = nonmakeup_seg_key
        self.ref_img_key = self.control_key

    def get_origin_img_input(self, batch, k, need_rearrange=True):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(self.device)
        if need_rearrange:
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_seg_img_input(self, batch, k):
        x = batch[k]
        x = x.to(self.device)
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

    def get_target(self, batch):
        makeup_img = self.get_origin_img_input(batch, self.makeup_img_key)
        nonmakeup_img = self.get_origin_img_input(batch, self.nonmakeup_img_key)
        makeup_seg = self.get_seg_img_input(batch, self.makeup_seg_key)
        nonmakeup_seg = self.get_seg_img_input(batch, self.nonmakeup_seg_key)
        target = self.teacher_model(makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg)
        target = target.clamp(-1, 1)
        return target

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        src_img = self.get_origin_img_input(batch, self.src_img_key)
        ref_img = self.get_origin_img_input(batch, self.ref_img_key)
        x = self.get_target(batch)

        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        txt_ctl = self.get_cond_txt_coding(batch)
        c_concat = torch.cat([src_img, ref_img], 1)
        return z, dict(c_crossattn=[txt_ctl], c_concat=[c_concat])

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, parameterization=self.parameterization)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, parameterization='eps', *args, **kwargs):
        if parameterization == 'eps':
            eps = super(BaseModel, self).apply_model(x_noisy, t, cond, *args, **kwargs)
            return eps
        elif parameterization == 'x0':
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
            return x_recon

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        src, ref = torch.chunk(c_cat, 2, dim=1)
        log["control_src"] = src * 2.0 - 1.0
        log["control_ref"] = ref * 2.0 - 1.0
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


class Teacher_IDT_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_target(self, batch):
        makeup_img = self.get_origin_img_input(batch, self.makeup_img_key)
        nonmakeup_img = self.get_origin_img_input(batch, self.nonmakeup_img_key)
        makeup_seg = self.get_seg_img_input(batch, self.makeup_seg_key)
        nonmakeup_seg = self.get_seg_img_input(batch, self.nonmakeup_seg_key)
        target = self.teacher_model(makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg)
        target = target.clamp(-1, 1)
        return target

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # 0: src+ref
        # 1: src+src
        # 2: ref+ref
        data_type = 0
        if self.training:
            data_type = torch.randint(0, 3, (1,)).item()
        if data_type == 0:
            src_img = self.get_origin_img_input(batch, self.src_img_key)
            ref_img = self.get_origin_img_input(batch, self.ref_img_key)
            x = self.get_target(batch)
        elif data_type == 1:
            src_img = self.get_origin_img_input(batch, self.src_img_key)
            ref_img = src_img
            x = self.get_origin_img_input(batch, self.nonmakeup_img_key)
        elif data_type == 2:
            ref_img = self.get_origin_img_input(batch, self.ref_img_key)
            src_img = ref_img
            x = self.get_origin_img_input(batch, self.makeup_img_key)
        else:
            assert 1 == 2, 'error...'
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        txt_ctl = self.get_cond_txt_coding(batch)
        c_concat = torch.cat([src_img, ref_img], 1)
        return z, dict(c_crossattn=[txt_ctl], c_concat=[c_concat])


class Teacher_IDT_Tmin_Model(Teacher_IDT_Model):
    def __init__(self, t_min, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_min = t_min

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(self.t_min, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)


# 0406 add
class Fixbackground(Teacher_IDT_Tmin_Model):
    def __init__(self, is_fixbkgrd, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_fixbkgrd = is_fixbkgrd

    def get_target(self, batch, is_fixbkgrd=False):
        makeup_img = self.get_origin_img_input(batch, self.makeup_img_key)
        nonmakeup_img = self.get_origin_img_input(batch, self.nonmakeup_img_key)
        makeup_seg = self.get_seg_img_input(batch, self.makeup_seg_key)
        nonmakeup_seg = self.get_seg_img_input(batch, self.nonmakeup_seg_key)
        target = self.teacher_model(makeup_img, nonmakeup_img, makeup_seg, nonmakeup_seg)
        if is_fixbkgrd:
            _bkgrd = (nonmakeup_seg == 0).float()    # background
            # _bkgrd += (nonmakeup_seg == 10).float()  # neck
            _bkgrd += (nonmakeup_seg == 11).float()  # teeth
            _bkgrd += (nonmakeup_seg == 12).float()  # hair
            _bkgrd = _bkgrd.unsqueeze(1)
            target = _bkgrd * ((nonmakeup_img + 1) / 2) + (1 - _bkgrd) * ((target + 1) / 2)
            target = target * 2.0 - 1.0
        target = target.clamp(-1, 1)
        return target

    def get_input(self, batch, k, bs=None, is_fixbkgrd=None, *args, **kwargs):
        if is_fixbkgrd is None:
            is_fixbkgrd = self.is_fixbkgrd
        # 0: src+ref
        # 1: src+src
        # 2: ref+ref
        data_type = 0
        if self.training:
            data_type = torch.randint(0, 3, (1,)).item()
        if data_type == 0:
            src_img = self.get_origin_img_input(batch, self.src_img_key)
            ref_img = self.get_origin_img_input(batch, self.ref_img_key)
            x = self.get_target(batch, is_fixbkgrd)
        elif data_type == 1:
            src_img = self.get_origin_img_input(batch, self.src_img_key)
            ref_img = src_img
            x = self.get_origin_img_input(batch, self.nonmakeup_img_key)
        elif data_type == 2:
            ref_img = self.get_origin_img_input(batch, self.ref_img_key)
            src_img = ref_img
            x = self.get_origin_img_input(batch, self.makeup_img_key)
        else:
            src_img = self.get_origin_img_input(batch, self.src_img_key)
            ref_img = self.get_origin_img_input(batch, self.ref_img_key)
            x = self.get_target(batch, is_fixbkgrd)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        txt_ctl = self.get_cond_txt_coding(batch)
        c_concat = torch.cat([src_img, ref_img], 1)
        return z, dict(c_crossattn=[txt_ctl], c_concat=[c_concat])

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        # todo
        z, c = self.get_input(batch, self.first_stage_key, bs=N, is_fixbkgrd=False)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        src, ref = torch.chunk(c_cat, 2, dim=1)
        log["control_src"] = src * 2.0 - 1.0
        log["control_ref"] = ref * 2.0 - 1.0
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
