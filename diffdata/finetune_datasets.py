import os
import lmdb
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LMDBDataset(Dataset):
    def __init__(self, path=None, transform=None, resolution=256):
        if path is None:
            path = os.path.join(os.environ['DATAROOT'], 'lmdbs', 'ffhq%dx%d' %(resolution, resolution))
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                ])
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(6)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        return img


class FFHQ256(LMDBDataset):
    def __getitem__(self, index):
        img = super().__getitem__(index)
        prompt = 'reconstruct'
        ref_img = img * 0
        src_img = (img + 1.0) / 2.0
        rt_dict = dict(jpg=img, txt=prompt, ref_img=ref_img, src_img=src_img)
        return rt_dict
