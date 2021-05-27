import os
import glob
import h5py
import argparse
import numpy as np
from PIL import Image


def generate_patches(image, label, mask, patch_per_img, patch_size=48):
    patches = np.zeros([image.shape[0], patch_per_img,
                        patch_size, patch_size], dtype=np.uint8)
    patches_label = np.zeros_like(patches)
    for i in range(image.shape[0]):
        j = 0
        while(j < patch_per_img):
            x = np.random.randint(
                0+patch_size//2, image.shape[2]-patch_size//2)
            y = np.random.randint(
                0+patch_size//2, image.shape[1]-patch_size//2)
            if not check(mask[i], x, y, patch_size):
                continue
            patches[i, j, :, :] = image[i, y-patch_size//2:y +
                                        patch_size//2, x-patch_size//2:x+patch_size//2]
            patches_label[i, j, :, :] = label[i, y-patch_size //
                                              2:y+patch_size//2, x-patch_size//2:x+patch_size//2]
            j += 1
    return patches, patches_label


def check(mask, x, y, patch_size):
    '''检查随机生成的patch是否完全位于mask的范围内'''
    mask[mask != 0] = 1
    region = mask[y-patch_size//2:y+patch_size //
                  2, x-patch_size//2:x+patch_size//2]
    region = region.reshape(-1)
    if region.sum() == len(region):
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_per_img', type=int, default=400)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=1)
    config = parser.parse_args()

    np.random.seed(config.seed)
    os.makedirs(config.data_path, exist_ok=True)
    image_list = glob.glob('training/images/*')
    label_list = glob.glob('training/1st_manual/*')
    mask_list = glob.glob('training/mask/*')
    image = np.zeros([len(image_list), 512, 512], dtype=np.uint8)
    label = np.zeros([len(label_list), 512, 512], dtype=np.uint8)
    mask = np.zeros([len(mask_list), 512, 512], dtype=np.uint8)
    for i in range(len(image_list)):
        pil_img = Image.open(image_list[i]).convert("L").resize([512, 512])
        pil_label = Image.open(label_list[i]).resize([512, 512])
        pil_mask = Image.open(mask_list[i]).resize([512, 512])
        image[i, :, :] = np.array(pil_img)
        label[i, :, :] = np.array(pil_label)
        mask[i, :, :] = np.array(pil_mask)
    label[label != 0] = 255
    mask[mask != 0] = 255

    # 生成patch
    patches, patches_label = generate_patches(
        image, label, mask, config.patch_per_img, config.patch_size)
    train_data = patches[:, :int(config.patch_per_img*0.8), :, :]
    train_mask = patches_label[:, :int(config.patch_per_img*0.8), :, :]
    val_data = patches[:, int(config.patch_per_img*0.8):, :, :]
    val_mask = patches_label[:, int(config.patch_per_img*0.8):, :, :]

    # 保存数据
    with h5py.File('./data/DRIVE.h5', 'w') as f:
        f.create_dataset('train_data', data=train_data)
        f.create_dataset('train_mask', data=train_mask)
        f.create_dataset('val_data', data=val_data)
        f.create_dataset('val_mask', data=val_mask)
