import random
import os

import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data_yield(loader):
    while True:
        yield from loader


def load_data_inpa(
    *,
    gt_path=None,
    mask_path=None,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    return_dataloader=False,
    return_dict=False,
    max_len=None,
    drop_last=True,
    conf=None,
    offset=0,
    **kwargs
):
    """
    Load spectral patches from either `.npy` files or image directories.
    """

    if gt_path.endswith(".npy") and mask_path.endswith(".npy"):
        # ✅ If both gt and mask are .npy files, load them as NumPy datasets
        gt_data = np.load(gt_path)  # Ground truth
        mask_data = np.load(mask_path)  # Corresponding masks

        dataset = NpyDatasetInpa(
            gt_data,
            mask_data,
            image_size=image_size,
            random_crop=random_crop,
            random_flip=random_flip,
            return_dict=return_dict,
            max_len=max_len,
        )

    else:
        # ✅ Default behavior for loading images from directories
        gt_dir = os.path.expanduser(gt_path)
        mask_dir = os.path.expanduser(mask_path)

        gt_paths = _list_image_files_recursively(gt_dir)
        mask_paths = _list_image_files_recursively(mask_dir)

        assert len(gt_paths) == len(mask_paths)

        dataset = ImageDatasetInpa(
            image_size,
            gt_paths=gt_paths,
            mask_paths=mask_paths,
            random_crop=random_crop,
            random_flip=random_flip,
            return_dict=return_dict,
            max_len=max_len,
            conf=conf,
            offset=offset,
        )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=not deterministic, num_workers=1, drop_last=drop_last
    )

    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


def _list_image_files_recursively(data_dir):
    """
    Recursively list all image files in a directory.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1].lower()
        if ext in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class NpyDatasetInpa(Dataset):
    """
    Dataset class for handling .npy spectral patches for RePaint.
    """

    def __init__(self, gt_data, mask_data, image_size, random_crop=False, random_flip=True, return_dict=False, max_len=None):
        super().__init__()
        self.gt_data = gt_data  # Loaded spectral patches
        self.mask_data = mask_data  # Corresponding masks
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict
        self.max_len = max_len

    def __len__(self):
        return min(len(self.gt_data), self.max_len) if self.max_len else len(self.gt_data)

    def __getitem__(self, idx):
        arr_gt = self.gt_data[idx]  # Extract ground truth patch
        arr_mask = self.mask_data[idx]  # Extract corresponding mask

        if self.random_crop:
            arr_gt = random_crop_arr(arr_gt, self.image_size)
            arr_mask = random_crop_arr(arr_mask, self.image_size)
        else:
            arr_gt = center_crop_arr(arr_gt, self.image_size)
            arr_mask = center_crop_arr(arr_mask, self.image_size)

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_mask = arr_mask[:, ::-1]

        arr_gt = arr_gt.astype(np.float32)  # Keep as float32
        arr_mask = arr_mask.astype(np.float32)

        if len(arr_gt.shape) == 2:  # Ensure grayscale compatibility
            arr_gt = np.expand_dims(arr_gt, axis=0)
            arr_mask = np.expand_dims(arr_mask, axis=0)

        if self.return_dict:
            return {
                'GT': arr_gt,
                'GT_name': f"patch_{idx:05d}",
                'gt_keep_mask': arr_mask,
            }
        else:
            return arr_gt, arr_mask


class ImageDatasetInpa(Dataset):
    """
    Dataset class for handling image file inputs.
    """

    def __init__(
        self,
        resolution,
        gt_paths,
        mask_paths,
        random_crop=False,
        random_flip=True,
        return_dict=False,
        max_len=None,
        conf=None,
        offset=0,
    ):
        super().__init__()
        self.resolution = resolution

        gt_paths = sorted(gt_paths)[offset:]
        mask_paths = sorted(mask_paths)[offset:]

        self.local_gts = gt_paths[:max_len] if max_len else gt_paths
        self.local_masks = mask_paths[:max_len] if max_len else mask_paths

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict

    def __len__(self):
        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        mask_path = self.local_masks[idx]

        with bf.BlobFile(gt_path, "rb") as f:
            arr_gt = np.load(f)  # ✅ Load .npy instead of images

        with bf.BlobFile(mask_path, "rb") as f:
            arr_mask = np.load(f)  # ✅ Load .npy masks

        if self.random_crop:
            arr_gt = random_crop_arr(arr_gt, self.resolution)
            arr_mask = random_crop_arr(arr_mask, self.resolution)
        else:
            arr_gt = center_crop_arr(arr_gt, self.resolution)
            arr_mask = center_crop_arr(arr_mask, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_mask = arr_mask[:, ::-1]

        arr_gt = arr_gt.astype(np.float32)
        arr_mask = arr_mask.astype(np.float32)

        if len(arr_gt.shape) == 2:  # Ensure grayscale format
            arr_gt = np.expand_dims(arr_gt, axis=0)
            arr_mask = np.expand_dims(arr_mask, axis=0)

        if self.return_dict:
            return {
                'GT': arr_gt,
                'GT_name': os.path.basename(gt_path),
                'gt_keep_mask': arr_mask,
            }
        else:
            return arr_gt, arr_mask


def center_crop_arr(arr, image_size):
    """
    Center crops a NumPy array to the specified image size.
    """
    h, w = arr.shape
    crop_y = (h - image_size) // 2
    crop_x = (w - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(arr, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """
    Randomly crops a NumPy array to the specified image size.
    """
    h, w = arr.shape
    crop_size = random.randint(int(image_size / max_crop_frac), int(image_size / min_crop_frac))
    start_y = random.randint(0, h - crop_size)
    start_x = random.randint(0, w - crop_size)
    return arr[start_y : start_y + crop_size, start_x : start_x + crop_size]