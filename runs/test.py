import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from diffdata.datasets import TestFixed_Dataset
from pytorch_lightning import loggers as pl_loggers
from diffmk.logger import MakeupImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
# !!! modify begin !!!
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# model_yaml_path = './ablation_tmin_tmax.yaml'
model_yaml_path = './experiment/ablation_tmin_tmax.yaml'
log_root = '../../../results'
# log_root = '../../../../results'
batch_size = 1
sd_locked = True
only_mid_control = False
logger_freq = 200
learning_rate = 1e-5


model_path = './checkpoints/teacjer_scgan/latest-epoch.ckpt'

resume_path = model_path
# !!! model begin !!!
model = create_model(model_yaml_path).cpu()
# !!! model end   !!!

# !!! dataset begin !!!
dataset = TestFixed_Dataset()
dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=False)
print(len(dataset), len(dataset[0]), dataset[0].keys())
# !!! dataset end   !!!

# !!! logging begin !!!
log_dir = './0410/base/'
log_dir = os.path.join(log_dir, 'w_id_s-%.2f+w_id_r-%.2f+w_bk-%.2f+w_c_s-%.2f+w_c_r-%.2f+w_mkup-%.2f'
                       %(model.w_idt_src, model.w_idt_ref, model.w_bkgrd, model.w_cycle_content,
                         model.w_cycle_makeup, model.w_makeup))
# lambda_lip, lambda_eye, lambda_skin
log_dir = os.path.join(log_dir, 'l_lip-%d+l_eye-%f+l_skin-%.2f' %(model.lambda_lip, model.lambda_eye, model.lambda_skin))
log_dir = os.path.join(log_dir, 'tmin-%d+lr-%f+only_mid_contol-%s' %(model.t_min, learning_rate, str(model.only_mid_control)))
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

# !!! test begin !!!
saved_state_dict = load_state_dict(resume_path, location='cpu')
model.load_state_dict(saved_state_dict)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
trainer.test(model, dataloader)
# !!! test end   !!!

