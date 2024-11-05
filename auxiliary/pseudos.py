import os
import torch
import argparse
from PIL import Image
from ele_training.config import get_config
from ele_training.inference import Inference


class PseudoModel(Inference):
    def __init__(self, gpu_id):
        code_dir = os.environ['PSEUDO']
        config = get_config()
        parser = argparse.ArgumentParser("pseudo model...")
        args = parser.parse_args(args=[])
        args.name = 'load pseudo model'
        args.save_path = os.path.join(code_dir, 'result')
        # author: sow_pyramid_a5_e3d2_remapped.pth
        # ours 0318: A100_0318_epoch50_G.pth
        args.load_path = os.path.join(code_dir, 'ckpts/sow_pyramid_a5_e3d2_remapped.pth')
        # args.source_dir = os.path.join(code_dir, 'assets/images/non-makeup')
        # args.reference_dir = os.path.join(code_dir, 'assets/images/makeup')
        if gpu_id:
            args.gpu = str(gpu_id)
            args.gpu = 'cuda:' + args.gpu
            args.device = torch.device(args.gpu)
        else:
            args.device = 'cpu'
        super().__init__(config, args, args.load_path)

    def generate_pseudo_GT(self, src_path, ref_path):
        source = Image.open(src_path).convert('RGB')
        reference = Image.open(ref_path).convert('RGB')
        source_input, face, crop_face = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)
        if not (source_input and reference_input):
            return None

        source_input = self.prepare_input(*source_input)
        reference_input = self.prepare_input(*reference_input)
        PGT_SR, PGT_RS = self.solver.generate_makeup_GT(*source_input, *reference_input)
        return PGT_SR.squeeze(0), PGT_RS.squeeze(0)

def test():
    from torchvision.transforms import ToPILImage
    pseudo_model = PseudoModel(2)
    src_path = '../assets/non-makeup/source_1.png'
    ref_path = '../assets/makeup/reference_1.png'
    fake_A, fake_B = pseudo_model.generate_pseudo_GT(src_path, ref_path)
    A = ToPILImage()(fake_A.cpu())
    B = ToPILImage()(fake_B.cpu())
    A.show()
    B.show()
