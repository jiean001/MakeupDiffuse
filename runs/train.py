import os
import sys
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from diffdata.datasets import Ele_PGT_Dataset
from pytorch_lightning import loggers as pl_loggers
from diffmk.logger import MakeupImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# !!! modify begin !!!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_yaml_path = 'diffmodels-acmmm/base_diffusion_makeup.yaml'
log_root = '../../../results'
batch_size = 6
sd_locked = True
only_mid_control = False
logger_freq = 200
learning_rate = 1e-5
# !!! modify end  !!!

control_net_dir = os.environ["CONTROLNET"]
resume_path = os.path.join(control_net_dir, './models/control_sd15_ini.ckpt')
assert control_net_dir is not None, 'please set CONTROLNET'
# !!! model begin !!!
model = create_model(model_yaml_path).cpu()
# !!! model end   !!!

# !!! dataset begin !!!
dataset = Ele_PGT_Dataset()
dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True)
print(len(dataset), len(dataset[0]), dataset[0].keys())
# !!! dataset end   !!!

# !!! logging begin !!!
log_dir = './ACM-MM/0423'
log_dir = os.path.join(log_dir, 'w_id_s-%.2f+w_id_r-%.2f+w_bk-%.2f+w_c_s-%.2f+w_c_r-%.2f+w_mkup-%.2f'
                       %(model.w_idt_src, model.w_idt_ref, model.w_bkgrd, model.w_cycle_content,
                         model.w_cycle_makeup, model.w_makeup))
# lambda_lip, lambda_eye, lambda_skin
log_dir = os.path.join(log_dir, 'l_lip-%d+l_eye-%f+l_skin-%.2f' %(model.lambda_lip, model.lambda_eye, model.lambda_skin))
log_dir = os.path.join(log_dir, 'tmin-%d+tmax-%d+lr-%f+only_mid_contol-%s' %(model.t_min, model.t_max, learning_rate, str(model.only_mid_control)))
log_dir = os.path.join(log_dir, model.parameterization)
log_dir = os.path.join(log_dir, model.teacher_type)
log_dir = os.path.join(log_root, log_dir)
os.makedirs(log_dir, exist_ok=True)
base_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
image_logger = MakeupImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(log_dir, 'checkpoints'), every_n_train_steps=200)
# !!! logging end   !!!

# !!! trainer begin !!!
gpu_nums = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
trainer = pl.Trainer(gpus=1, precision=32, logger=base_logger, callbacks=[image_logger, checkpoint_callback])
# !!! trainer end   !!!

# !!! fit begin !!!
saved_state_dict = load_state_dict(resume_path, location='cpu')
# control_model.input_hint_block.0.weight [16, 3, 3, 3] -> [16, 6, 3, 3]
control_net_first_weight = saved_state_dict['control_model.input_hint_block.0.weight']
saved_state_dict['control_model.input_hint_block.0.weight'] = torch.cat([control_net_first_weight, control_net_first_weight], 1)
# add teacher ckpt to model
model_dict = model.state_dict()
for key in model_dict.keys():
    if key.startswith('teacher_model'):
        saved_state_dict[key] = model_dict[key]
model.load_state_dict(saved_state_dict)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
trainer.fit(model, dataloader)
# !!! fit end   !!!
