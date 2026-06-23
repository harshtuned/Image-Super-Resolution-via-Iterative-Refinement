import random
from io import BytesIO
import cv2
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import data.util as Util


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            # Load from lr folder directly; bicubic upscaling is done on the fly in __getitem__
            self.sr_path = Util.get_paths_from_images(
                '{}/lr_{}'.format(dataroot, l_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        # Robust 16-bit/8-bit grayscale loading helper (handles paths and bytes)
        def load_grayscale_tif(path_or_bytes):
            if isinstance(path_or_bytes, bytes):
                # Handle LMDB byte streams
                img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    img = np.array(Image.open(BytesIO(path_or_bytes)))
            else:
                # Handle standard file paths
                img = cv2.imread(path_or_bytes, cv2.IMREAD_UNCHANGED)
                if img is None:
                    img = np.array(Image.open(path_or_bytes))
            
            # Normalize 16-bit or float arrays to 8-bit [0, 255] safely
            if img.dtype == np.uint16 or img.dtype == np.float32 or img.dtype == np.float64:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                img = img.astype(np.uint8)
                
            return Image.fromarray(img)

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(self.r_res, str(index).zfill(5)).encode('utf-8'))
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8'))
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(self.l_res, str(index).zfill(5)).encode('utf-8'))
                
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(self.r_res, str(new_index).zfill(5)).encode('utf-8'))
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8'))
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(self.l_res, str(new_index).zfill(5)).encode('utf-8'))
                
                img_HR = load_grayscale_tif(hr_img_bytes)
                img_SR = load_grayscale_tif(sr_img_bytes)
                img_SR = img_SR.resize((self.r_res, self.r_res), Image.BICUBIC)
                if self.need_LR:
                    img_LR = load_grayscale_tif(lr_img_bytes)
        else:
            # Load images using the path-based helper
            img_HR = load_grayscale_tif(self.hr_path[index])
            img_SR = load_grayscale_tif(self.sr_path[index])
            img_SR = img_SR.resize((self.r_res, self.r_res), Image.BICUBIC)
            if self.need_LR:
                img_LR = load_grayscale_tif(self.lr_path[index])

        # Apply data augmentations and transformations
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}