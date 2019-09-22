import os
import numpy as np
from datetime import datetime

import torch
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms
from model import VAE

from torch.utils.tensorboard import SummaryWriter

import my_mnist as mn

# mnist directory
file_dir = 'C:\\workspace\\dataset\\MNIST\\'

def train():
    # データ読み込み & 変形
    images = mn.load_mnist_img(file_dir, mode='train')
    labels = mn.load_mnist_labels(file_dir, mode='train')
    onehot_labels = mn.mnist_labels_to_onehot(labels)
    flat_images = mn.mnist_images_to_vector(images)

    images = torch.from_numpy(flat_images)
    labels = torch.from_numpy(onehot_labels)

    dataset_train = \
        data.TensorDataset(images, labels)
    dataloader_train = \
        data.DataLoader(dataset=dataset_train, batch_size=1000, shuffle=True, num_workers=4)

    # モデル初期設定
    num_epochs = 50
    Latent_num = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE(Latent_num).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)

    # log
    loss_log = []
    writer = SummaryWriter(log_dir='logs')

    # 学習
    model.train()

    for i in range(num_epochs):
        losses = []
        for x, t in dataloader_train:
            x = x.to(device)
            model.zero_grad()
            y = model(x)
            loss = model.loss(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            loss_log.append(loss.cpu().detach().numpy())
        print("EPOCH: {} loss: {}".format(i, np.average(losses)))
        writer.add_scalar("train_loss", np.average(losses), i)
    writer.close()

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    # modelとloss遷移の保存
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    np.savez('./logs/loss_VAE' + time + '.npz', loss=np.array(loss_log))
    torch.save(model.state_dict(), './saved_model/VAE' + time + '.pth')

if __name__ == '__main__':
    train()