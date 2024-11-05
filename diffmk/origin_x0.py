from cldm.cldm import *
from ldm.util import default


class BaseModel(ControlLDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ModifiedX0(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            eps = super(ModifiedX0, self).apply_model(x_noisy, t, cond, *args, **kwargs)
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
