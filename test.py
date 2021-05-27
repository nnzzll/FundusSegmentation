import glob
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from networks import UNet, UNetPlusPlus, UNetPlusPlus_L1, UNetPlusPlus_L2, UNetPlusPlus_L3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='UNet')
    parser.add_argument('--epochs', type=int, default=20)
    config = parser.parse_args()
    patch_size = config.patch_size
    test_img_list = glob.glob('test/images/*')
    mask_list = glob.glob('test/mask/*')
    state = torch.load(
        'models/{}-epoch{}.pth'.format(config.model, config.epochs), map_location='cpu')
    if config.model == 'UNet':
        model = UNet(1, 1)
    elif config.model == 'UNet++':
        model = UNetPlusPlus(1, 1)
    elif config.model == 'UNet++L1':
        model = UNetPlusPlus_L1(1, 1)
    elif config.model == 'UNet++L2':
        model = UNetPlusPlus_L2(1, 1)
    elif config.model == 'UNet++L3':
        model = UNetPlusPlus_L3(1, 1)
    else:
        raise NotImplementedError
    model.load_state_dict(state['net'])
    model.to("cuda")
    model.eval()

    with torch.no_grad():
        for i in range(len(test_img_list)):
            image = Image.open(test_img_list[i]).convert(
                "L").resize([512, 512])
            mask = Image.open(mask_list[i]).resize([512, 512])
            image = np.array(image)/255
            mask = np.array(mask)/255
            num_patch = 512//patch_size
            preds = np.zeros_like(image)
            for m in range(num_patch):
                for n in range(num_patch):
                    region = image[m*patch_size:(m+1)*patch_size,
                                   n*patch_size:(n+1)*patch_size]
                    inputs = torch.Tensor(region).view(
                        1, 1, patch_size, patch_size).to("cuda")
                    preds[m*patch_size:(m+1)*patch_size, n*patch_size:(n+1)*patch_size] = model(inputs,
                                                                                                torch.sigmoid).squeeze().cpu().numpy()
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image, 'gray')
            ax[1].imshow(preds*mask, 'gray')
            plt.show()
