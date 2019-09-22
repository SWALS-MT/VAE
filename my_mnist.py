import gzip
import numpy as np
import cv2


def load_mnist_img(file_dir, mode='train'):
    if mode == 'train':
        file_path = file_dir + 'train-images-idx3-ubyte.gz'
    elif mode == 'eval':
        file_path = file_dir + 't10k-images-idx3-ubyte.gz'
    else:
        print('mode error')
        return None
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, 784)
    data = data.reshape(data.shape[0], 28, 28)
    data = data.astype(np.float32) / 255.0

    return data

def load_mnist_labels(file_dir, mode='train'):
    if mode == 'train':
        file_path = file_dir + 'train-labels-idx1-ubyte.gz'
    elif mode == 'eval':
        file_path = file_dir + 't10k-labels-idx1-ubyte.gz'
    else:
        print('mode error')
        return None
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data

def mnist_images_to_vector(images):
    images = images.reshape(images.shape[0], 784)
    return images

def mnist_labels_to_onehot(labels):
    one_hot = np.identity(10)[labels].astype(np.float32)

    return one_hot

if __name__ == '__main__':
    file_dir = 'C:\\workspace\\dataset\\MNIST\\'
    imgs = load_mnist_img(file_dir, mode='train')
    labels = load_mnist_labels(file_dir, mode='train')
    one_hot = mnist_labels_to_onehot(labels)
    print(imgs.dtype)
    print(one_hot.dtype)