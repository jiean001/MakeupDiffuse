from diffmk.makeup_teacher import *


class FinetuneModelFFHQ(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_origin_img_input(self, batch, k, need_rearrange=False):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(self.device)
        if need_rearrange:
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_target(self, batch):
        target = self.get_origin_img_input(batch, self.first_stage_key, need_rearrange=False)
        return target
