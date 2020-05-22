def load_data_fashion_mnist(batch_size, root='fashion-mnist/', resize=None):
    from mxnet.gluon import data as gdata
    import sys
    transformer = [gdata.vision.transforms.Resize(resize)] if resize else []
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(
        train=True, root=root).transform_first(transformer)
    mnist_test = gdata.vision.FashionMNIST(
        train=False, root=root).transform_first(transformer)
    num_works = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=num_works)
    test_iter = gdata.DataLoader(
        mnist_test, batch_size, shuffle=True, num_workers=num_works)
    return train_iter, test_iter