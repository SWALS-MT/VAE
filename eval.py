import os
import numpy as np
from datetime import datetime
import cv2

import torch
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms
from model import VAE

from torch.utils.tensorboard import SummaryWriter

import my_mnist as mn

# mnist directory
file_dir = 'C:\\workspace\\dataset\\MNIST\\'

cv2.namedWindow('img_sample', cv2.WINDOW_NORMAL)
cv2.namedWindow('out_sample', cv2.WINDOW_NORMAL)

def eval():
    # データ読み込み & 変形
    images = mn.load_mnist_img(file_dir, mode='eval')
    labels = mn.load_mnist_labels(file_dir, mode='eval')
    onehot_labels = mn.mnist_labels_to_onehot(labels)
    flat_images = mn.mnist_images_to_vector(images)

    images = torch.from_numpy(flat_images)
    labels = torch.from_numpy(onehot_labels)

    # モデル初期設定
    Latent_num = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE(Latent_num)
    model.load_state_dict(torch.load('saved_model/VAE2019_09_22_01_30_17.pth'))
    model.to(device)

    print(model)

    model.eval()

    for i in range(labels.size()[0]):
        img = images[i:i+1].to(device)
        img_sample = tensor_to_display_image(img)

        print(img)
        with torch.no_grad():
            out, z = model(img)
        out_sample = tensor_to_display_image(out)

        cv2.imshow('img_sample', img_sample)
        cv2.imshow('out_sample', out_sample)
        key = cv2.waitKey(500)
        if key == 27:
            break


def tensor_to_display_image(array):
    array = array.to('cpu').detach().numpy()[0]
    array = array.reshape(28, 28)
    array = array * 255
    array[array > 255] = 255
    array[array < 0] = 0
    array = array.astype('uint8')
    return array

if __name__ == '__main__':
    eval()