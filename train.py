import h5py
import time
import argparse
import torch
import torch.optim as optim
from torch.backends import cudnn
from networks import UNet, UNetPlusPlus, UNetPlusPlus_L1, UNetPlusPlus_L2, UNetPlusPlus_L3, print_param
from metric import DiceLoss
from metric import Dice, Jaccard, pixel_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_per_img', type=int, default=400)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='./data/DRIVE.h5')
    parser.add_argument('--model', type=str, default='UNet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    config = parser.parse_args()

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True
    with h5py.File(config.data_path, 'r') as f:
        train_data = torch.Tensor(
            f['train_data'][:].reshape(-1, config.patch_size, config.patch_size)/255)
        train_mask = torch.Tensor(
            f['train_mask'][:].reshape(-1, config.patch_size, config.patch_size)/255)
        val_data = torch.Tensor(
            f['val_data'][:].reshape(-1, config.patch_size, config.patch_size)/255)
        val_mask = torch.Tensor(
            f['val_mask'][:].reshape(-1, config.patch_size, config.patch_size)/255)
    train_data = train_data.unsqueeze(1)
    train_mask = train_mask.unsqueeze(1)
    batch_size = config.batch_size
    n_batch = len(train_data)/batch_size
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
    print_param(model, config.model)
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = DiceLoss()
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0
        epoch_dcs = 0
        epoch_jac = 0
        epoch_acc = 0
        begin = time.time()
        i = 0
        for step in range(int(n_batch)):
            inputs, targets = train_data[i:i+batch_size].to(
                'cuda'), train_mask[i:i+batch_size].to('cuda')
            optimizer.zero_grad()
            output = model(inputs, torch.sigmoid)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_dcs += Dice(output, targets)
            epoch_jac += Jaccard(output, targets)
            epoch_acc += pixel_accuracy(output, targets)
            i += batch_size
            print("Batch No.{}/{}".format(step+1, n_batch), end='\r')
        epoch_loss /= n_batch
        epoch_dcs /= n_batch
        epoch_jac /= n_batch
        epoch_acc /= n_batch
        print("Epoch:{}/{} Loss:{} Dice:{:.2f}% Jaccard:{:.2f}% Acc:{:.2f}% Time:{:.3f}s".format(
            epoch+1, config.epochs, epoch_loss, epoch_dcs*100, epoch_jac*100, epoch_acc*100, time.time()-begin))

    state = {'net': model.state_dict()}
    torch.save(
        state, './models/{}-epoch{}.pth'.format(config.model, config.epochs))
    print("训练结束")
