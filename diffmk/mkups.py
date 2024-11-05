import os
import torch
import numpy as np
from cldm.cldm import *
from cldm.histogram_matching import *
from cldm.cddim import MKDDIMSampler
from cldm.utils import get_grid_image
from PIL import Image


class OnlyCycle(ControlLDM):
    def __init__(self, src_img_key, src_inv_key, ref_img_key, ref_inv_key, iter_finetune, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_img_key = src_img_key
        self.src_inv_key = src_inv_key
        self.ref_img_key = ref_img_key
        self.ref_inv_key = ref_inv_key
        self.saved_root = '../../mkup_dataset/debug/'

        self.t0 = 100
        self.ddim_sampler = None
        self.iter_finetune = iter_finetune

    def on_fit_start(self) -> None:
        self.update_schedule()
        self.ddim_sampler = MKDDIMSampler(self)
        self.ddim_sampler.make_schedule(ddim_num_steps=self.iter_finetune)

    def update_schedule(self):
        self.register_schedule(given_betas=None, beta_schedule='linear', timesteps=self.t0,
                               linear_start=self.linear_start, linear_end=self.linear_end, cosine_s=8e-3)

    # def training_step(self, batch, batch_idx):
    #     pass

    def get_origin_img_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(self.device)
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

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        src_img = self.get_origin_img_input(batch, self.src_img_key)
        ref_img = self.get_origin_img_input(batch, self.ref_img_key)
        txt_ctl = self.get_cond_txt_coding(batch)
        src_inv = batch[self.src_inv_key]
        ref_inv = batch[self.ref_inv_key]
        return src_inv, ref_inv, dict(c_crossattn=[txt_ctl], c_concat_s=[src_img],  c_concat_r=[ref_img])

    def shared_step(self, batch, **kwargs):
        src_inv, ref_inv, c = self.get_input(batch, self.first_stage_key)
        loss = self(src_inv, ref_inv, c)
        return loss

    def forward(self, src_inv, ref_inv, c, *args, **kwargs):
        t = torch.randint(self.iter_finetune, self.iter_finetune+1, (src_inv.shape[0],), device=self.device).long()
        return self.p_losses(src_inv, ref_inv, c, t, *args, **kwargs)

    def p_losses(self, src_inv, ref_inv, c, t, noise=None):
        # generate Sr=src+ref
        # c_crossattn c_concat
        c['c_concat'] = c['c_concat_r']
        src_ref_z_gen = self.ddim_sampler.reconstruct(x_latent=src_inv, cond=c, t_start=self.iter_finetune)
        # range [-1, 1]
        rec_src_ref = self.decode_first_stage(src_ref_z_gen)
        # range: [0, 1]
        rec_src_ref = (rec_src_ref + 1.0) / 2.0
        # todo 1... ID loss + style loss + perception loss [Source]

        # generate Rs=ref+src
        c['c_concat'] = c['c_concat_s']
        ref_src_z_gen = self.ddim_sampler.reconstruct(x_latent=ref_inv, cond=c, t_start=self.iter_finetune)
        rec_ref_src = self.decode_first_stage(ref_src_z_gen)
        rec_ref_src = (rec_ref_src + 1.0) / 2.0
        # todo 2... ID loss + style loss + perception loss [Target]

        # Cycle Ss=src+Rs
        c['c_concat'] = [rec_ref_src.detach()]
        src_src_z_gen = self.ddim_sampler.reconstruct(x_latent=src_inv, cond=c, t_start=self.iter_finetune)
        # rec_src_src = self.decode_first_stage(src_src_z_gen)
        # rec_src_src = (rec_src_src + 1.0) / 2.0

        # Cycle Rr=ref+Sr
        c['c_concat'] = [rec_src_ref.detach()]
        ref_ref_z_gen = self.ddim_sampler.reconstruct(x_latent=ref_inv, cond=c, t_start=self.iter_finetune)
        # rec_ref_ref = self.decode_first_stage(ref_ref_z_gen)
        # rec_ref_ref = (rec_ref_ref + 1.0) / 2.0

        # target_src_encode = torch.cat(c['c_concat_s'], 0)
        # target_ref_encode = torch.cat(c['c_concat_r'], 0)

        src_encoder_posterior = self.encode_first_stage(torch.cat(c['c_concat_s'], 0) * 2.0 - 1.0)
        ref_encoder_posterior = self.encode_first_stage(torch.cat(c['c_concat_r'], 0) * 2.0 - 1.0)
        target_src_encode = self.get_first_stage_encoding(src_encoder_posterior).detach()
        target_ref_encode = self.get_first_stage_encoding(ref_encoder_posterior).detach()


        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        loss_cycle_src = self.get_loss(target_src_encode, src_src_z_gen, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_cycle_src': loss_cycle_src.mean()})
        loss_cycle_ref = self.get_loss(target_ref_encode, ref_ref_z_gen, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_cycle_ref': loss_cycle_ref.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = ((loss_cycle_src + loss_cycle_ref) * 0.5) / torch.exp(logvar_t) + logvar_t
        # todo check l_simple_weight
        loss = self.l_simple_weight * loss.mean()

        # todo check loss_vlb
        # loss_vlb = self.get_loss(rec_src_src, target_src, mean=False).mean(dim=(1, 2, 3))
        # loss_vlb += self.get_loss(rec_ref_ref, target_ref, mean=False).mean(dim=(1, 2, 3))
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb * 0.5).mean()
        # loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        # loss += (self.original_elbo_weight * loss_vlb)
        # loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        return eps

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        log = dict()
        src_inv, ref_inv, c = self.get_input(batch, self.first_stage_key)

        c['c_concat'] = c['c_concat_r']
        src_ref_z_gen = self.ddim_sampler.reconstruct(x_latent=src_inv, cond=c, t_start=self.iter_finetune)
        rec_src_ref = self.decode_first_stage(src_ref_z_gen)

        c['c_concat'] = c['c_concat_s']
        ref_src_z_gen = self.ddim_sampler.reconstruct(x_latent=ref_inv, cond=c, t_start=self.iter_finetune)
        rec_ref_src = self.decode_first_stage(ref_src_z_gen)

        log["rec_src_ref"] = rec_src_ref  # get_grid_image(rec_src_ref)
        log["rec_ref_src"] = rec_ref_src  # get_grid_image(rec_ref_src)
        log["ori_src"] = torch.cat(c['c_concat_s'], 0) * 2.0 - 1.0  # get_grid_image(torch.cat(c['c_concat_s'], 0), is_scale=False)
        log["ori_ref"] = torch.cat(c['c_concat_r'], 0) * 2.0 - 1.0  # get_grid_image(torch.cat(c['c_concat_r'], 0), is_scale=False)
        return log

    # ref decode_first_stage
    def decode_latent_code(self, z, predict_cids=False, force_not_quantize=False):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)


class CycleMakeupModel(OnlyCycle):
    def __init__(self, src_msk_key, ref_msk_key, weight_loss_cycle, weight_loss_makeup, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_msk_key = src_msk_key
        self.ref_msk_key = ref_msk_key
        self.weight_loss_cycle = weight_loss_cycle
        self.weight_loss_makeup = weight_loss_makeup
        self.lambda_his_lip = 1.0
        self.lambda_his_skin_1 = 1.0
        self.lambda_his_skin_2 = 1.0
        self.lambda_his_eye = 1.0
        self.criterionL1 = torch.nn.L1Loss()

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        src_inv, ref_inv, c = super(CycleMakeupModel, self).get_input(batch, k)
        src_msk = self.get_origin_img_input(batch, self.src_msk_key)
        ref_msk = self.get_origin_img_input(batch, self.ref_msk_key)
        # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
        # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
        return src_inv, ref_inv, src_msk, ref_msk, c

    def shared_step(self, batch, **kwargs):
        src_inv, ref_inv, src_msk, ref_msk, c = self.get_input(batch, self.first_stage_key)
        loss = self(src_inv, ref_inv, src_msk, ref_msk, c)
        return loss

    def forward(self, src_inv, ref_inv, src_msk, ref_msk, c, *args, **kwargs):
        t = torch.randint(self.iter_finetune, self.iter_finetune+1, (src_inv.shape[0],), device=self.device).long()
        return self.p_losses(src_inv, ref_inv, src_msk, ref_msk, c, t, *args, **kwargs)
    
    def generate_image(self, inv, c, c_replace=None, c_type='c_concat_r'):
        if c_replace:
            c['c_concat'] = c_replace
        else:
            c['c_concat'] = c[c_type]
        src_ref_z_gen = self.ddim_sampler.reconstruct(x_latent=inv, cond=c, t_start=self.iter_finetune)
        rec_src_ref = self.decode_latent_code(src_ref_z_gen)
        rec_src_ref = (rec_src_ref + 1.0) / 2.0
        return rec_src_ref.clamp(0, 1)
    
    def p_loss_hist_lip(self, SR, RS, src_msk, ref_msk, S, R):
        m_S_lip, m_R_lip, idx_S_lip, idx_R_lip = self.get_msk_lip(src_msk, ref_msk)
        sr_lip_loss_his = self.criterionHis(SR, R, m_S_lip, m_R_lip, idx_S_lip) * self.lambda_his_lip
        rs_lip_loss_his = self.criterionHis(RS, S, m_R_lip, m_S_lip, idx_R_lip) * self.lambda_his_lip
        return sr_lip_loss_his, rs_lip_loss_his
    
    def p_loss_hist_skin(self, SR, RS, src_msk, ref_msk, S, R):
        m_S_skin, m_R_skin, idx_S_skin, idx_R_skin = self.get_msk_skin(src_msk, ref_msk)
        sr_skin_loss_his = self.criterionHis(SR, R, m_S_skin, m_R_skin, idx_S_skin) * self.lambda_his_skin_1
        rs_skin_loss_his = self.criterionHis(RS, S, m_R_skin, m_S_skin, idx_R_skin) * self.lambda_his_skin_2
        return sr_skin_loss_his, rs_skin_loss_his
    
    def p_loss_hist_eye(self, SR, RS, src_msk, ref_msk, S, R):
        m_S_eye_left, m_R_eye_left, idx_S_eye_left, idx_R_eye_left, m_S_eye_right, m_R_eye_right, idx_S_eye_right, idx_R_eye_right = self.get_msk_eye(src_msk, ref_msk)
        sr_eye_left_loss_his = self.criterionHis(SR, R, m_S_eye_left, m_R_eye_left, idx_S_eye_left) * self.lambda_his_eye
        rs_eye_left_loss_his = self.criterionHis(RS, S, m_R_eye_left, m_S_eye_left, idx_R_eye_left) * self.lambda_his_eye
        sr_eye_right_loss_his = self.criterionHis(SR, R, m_S_eye_right, m_R_eye_right, idx_S_eye_right) * self.lambda_his_eye
        rs_eye_right_loss_his = self.criterionHis(RS, S, m_R_eye_right, m_S_eye_right, idx_R_eye_right) * self.lambda_his_eye
        return sr_eye_left_loss_his, rs_eye_left_loss_his, sr_eye_right_loss_his, rs_eye_right_loss_his
    
    def p_loss_makeup(self, SR, RS, src_msk, ref_msk, S, R):
        sr_lip_loss_his, rs_lip_loss_his = self.p_loss_hist_lip(SR, RS, src_msk, ref_msk, S, R)
        sr_skin_loss_his, rs_skin_loss_his = self.p_loss_hist_skin(SR, RS, src_msk, ref_msk, S, R)
        sr_eye_left_loss_his, rs_eye_left_loss_his, sr_eye_right_loss_his, rs_eye_right_loss_his = self.p_loss_hist_eye(SR, RS, src_msk, ref_msk, S, R)
        loss_makeup = (sr_lip_loss_his + rs_lip_loss_his) + (sr_skin_loss_his + sr_skin_loss_his)
        loss_makeup += (sr_eye_left_loss_his + rs_eye_left_loss_his + sr_eye_right_loss_his + rs_eye_right_loss_his)
        return loss_makeup

    def p_losses(self, src_inv, ref_inv, src_msk, ref_msk, c, t, noise=None):
        rec_src_ref = self.generate_image(src_inv, c, c_type='c_concat_r')
        rec_ref_src = self.generate_image(ref_inv, c, c_type='c_concat_s')
        rec_src_src = self.generate_image(src_inv, c, c_replace=[rec_ref_src.detach()])
        rec_ref_ref = self.generate_image(ref_inv, c, c_replace=[rec_src_ref.detach()])
        
        target_src_src = torch.cat(c['c_concat_s'], 0).detach()
        target_ref_ref = torch.cat(c['c_concat_r'], 0).detach()
    
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        loss_cycle_src = self.get_loss(target_src_src, rec_src_src, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_cycle_src': loss_cycle_src.mean()})
        loss_cycle_ref = self.get_loss(target_ref_ref, rec_ref_ref, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_cycle_ref': loss_cycle_ref.mean()})
        loss_makeup = self.p_loss_makeup(SR=rec_src_ref, RS=rec_ref_src, src_msk=src_msk, ref_msk=ref_msk, S=target_src_src, R=target_ref_ref)
        loss_dict.update({f'{prefix}/makeup_loss': loss_makeup.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = ((loss_cycle_src + loss_cycle_ref) * 0.5) / torch.exp(logvar_t) + logvar_t
        # todo check l_simple_weight
        loss = self.l_simple_weight * loss.mean()
        loss = loss + self.weight_loss_makeup * loss_makeup
        return loss, loss_dict

    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2

    def get_msk_lip(self, mask_A, mask_B):
        mask_A_lip = (mask_A == 7).float() + (mask_A == 9).float()
        mask_B_lip = (mask_B == 7).float() + (mask_B == 9).float()
        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
        return mask_A_lip, mask_B_lip, index_A_lip, index_B_lip

    def get_msk_skin(self, mask_A, mask_B):
        mask_A_skin = (mask_A==1).float() + (mask_A==6).float() + (mask_A==13).float()
        mask_B_skin = (mask_B==1).float() + (mask_B==6).float() + (mask_B==13).float()
        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
        return mask_A_skin, mask_B_skin, index_A_skin, index_B_skin
    
    def get_msk_eye(self, mask_A, mask_B):
        mask_A_eye_left = (mask_A==4).float()
        mask_A_eye_right = (mask_A==5).float()
        mask_B_eye_left = (mask_B==4).float()
        mask_B_eye_right = (mask_B==5).float()
        mask_A_face = (mask_A==1).float() + (mask_A==6).float()
        mask_B_face = (mask_B==1).float() + (mask_B==6).float()
        # avoid the situation that images with eye closed
        # if not ((mask_A_eye_left>0).any() and (mask_B_eye_left>0).any() and (mask_A_eye_right > 0).any() and (mask_B_eye_right > 0).any()):
        #     continue
        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)
        return mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left, mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right
    
    def rebound_box(self, mask_A, mask_B, mask_A_face):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                            mask_A_face[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
        mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                            mask_A_face[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
        return mask_A_temp, mask_B_temp
    
    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (input_data * 255).squeeze()
        target_data = (target_data * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        # dstImg = (input_masked.data).cpu().clone()
        # refImg = (target_masked.data).cpu().clone()
        input_match = histogram_matching(input_masked, target_masked, index)
        # input_match = self.to_var(input_match, requires_grad=False)
        # todo detach
        loss = self.criterionL1(input_masked, input_match)
        return loss

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        log = dict()
        # src_inv, ref_inv, src_msk, ref_msk, c
        src_inv, ref_inv, src_msk, ref_msk, c = self.get_input(batch, self.first_stage_key)

        c['c_concat'] = c['c_concat_r']
        src_ref_z_gen = self.ddim_sampler.reconstruct(x_latent=src_inv, cond=c, t_start=self.iter_finetune)
        rec_src_ref = self.decode_first_stage(src_ref_z_gen)

        c['c_concat'] = c['c_concat_s']
        ref_src_z_gen = self.ddim_sampler.reconstruct(x_latent=ref_inv, cond=c, t_start=self.iter_finetune)
        rec_ref_src = self.decode_first_stage(ref_src_z_gen)

        log["rec_src_ref"] = rec_src_ref
        log["rec_ref_src"] = rec_ref_src
        log["ori_src"] = torch.cat(c['c_concat_s'], 0) * 2.0 - 1.0
        log["ori_ref"] = torch.cat(c['c_concat_r'], 0) * 2.0 - 1.0
        return log


class OnlyMakeupModel(CycleMakeupModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def p_losses(self, src_inv, ref_inv, src_msk, ref_msk, c, t, noise=None):
        rec_src_ref = self.generate_image(src_inv, c, c_type='c_concat_r')
        rec_ref_src = self.generate_image(ref_inv, c, c_type='c_concat_s')

        target_src_src = torch.cat(c['c_concat_s'], 0).detach()
        target_ref_ref = torch.cat(c['c_concat_r'], 0).detach()

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        loss_makeup = self.p_loss_makeup(SR=rec_src_ref, RS=rec_ref_src, src_msk=src_msk, ref_msk=ref_msk, S=target_src_src, R=target_ref_ref)
        loss_dict.update({f'{prefix}/makeup_loss': loss_makeup.mean()})
        loss = self.weight_loss_makeup * loss_makeup
        return loss, loss_dict