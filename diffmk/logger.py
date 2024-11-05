from cldm.logger import *


class MakeupImageLogger(ImageLogger):
    def __init__(self, *args, **kwargs):
        super(MakeupImageLogger, self).__init__(*args, **kwargs)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            # pass
            self.log_img(pl_module, batch, batch_idx, split="train")
