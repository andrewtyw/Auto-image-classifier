import sys
import os

# root_path = os.path.abspath(__file__)
# root_path = '/'.join(root_path.split('/')[:-2])

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

import torch.nn as nn
import torch
from torchvision import transforms, datasets
import pickle
from resnet import resnet34, resnet101, save, load
from PIL import Image
import time
import numpy as np
import heapq
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm


def img_filter(path):
    print("正在检查图片是否合法")
    child_folder = os.listdir(path)
    for child in child_folder:
        img_names = os.listdir(os.path.join(path, child))
        for img_name in img_names:
            img_path = os.path.join(os.path.join(path, child), img_name)
            try:
                _ = Image.open(img_path)
            except:
                os.remove(img_path)
                print("{} invalid, remove!".format(img_name))


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           augment=True,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """

        Params
        ------
        - data_dir: 数据集的地址.
        - batch_size: 就如名字一样.
        - augment: 是否图像增强.
        - random_seed: 随机种子.
        - valid_size: 验证集比例.
        - shuffle: 是否打乱.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        Returns
        -------
        - train_loader: training set iterator.
        - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transforms
    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize])
    if augment:
        train_transform = transforms.Compose([transforms.RandomRotation((-20, 20)),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
    else:
        train_transform = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize])

    # load the dataset
    # 相同的数据, 但是不同的transform
    try:
        train_dataset = datasets.ImageFolder(
            root=data_dir, transform=train_transform,
        )
        label2name = train_dataset.class_to_idx
        valid_dataset = datasets.ImageFolder(
            root=data_dir, transform=valid_transform,
        )
    except:
        print("图片数量不够, 请运行fetch继续下载")

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader, label2name, len(train_idx), len(valid_idx))


def adjust_learning_rate(optimizer, args, cur_epoch, total_epoch):
    """学习率衰减"""
    # print("cur_epoch,total_epoch", cur_epoch, total_epoch)
    # print(cur_epoch / total_epoch)
    lr = args.max_lr - (cur_epoch / total_epoch) * (args.max_lr - args.min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fish(args):
    img_filter(args.data_path)
    model_dict = {'resnet101': resnet101, 'resnet34': resnet34}
    device = torch.device("cuda:{}".format(args.cuda_index))

    (train_loader, valid_loader, label2name, train_num, val_num) = get_train_valid_loader(args.data_path,
                                                                                          args.batch_size, 16,
                                                                                          valid_size=0.2, num_workers=0,
                                                                                          pin_memory=False)
    save(label2name, os.path.join(args.save_dir, "label2name_{}.pkl".format(args.model_name)))
    model = model_dict[args.model_name]()
    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(os.path.join(args.pretrain_model_path, args.model_name + ".pth")), strict=False)
    except Exception:
        print(Exception)
        print("预训练模型加载失败, 将不采用预训练权重")
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, len(label2name))
    model.to(device)
    # print(model)
    batch = next(iter(train_loader))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    debug_stop = 1
    trainloss = []
    train_ac = []
    vail_ac = []
    best_acc = -1
    for epoch in range(args.epoch):
        lr = adjust_learning_rate(optimizer, args, epoch, args.epoch)

        # train
        model.train()
        running_loss = 0.0
        train_acc = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_predict_y = torch.max(logits, dim=1)[1]
            train_acc += (train_predict_y == labels.to(device)).sum().item()
            # lr_scheduler.step()#衰减学习率
            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "=" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            trainloss.append(loss)
            print(
                "\rlr: {:.8f} train loss: {:^3.0f}%[{}->{}]{:.4f} step: {}".format(lr, int(rate * 100), a, b, loss,
                                                                                   step),
                end="")
        print()
        train_accurate = train_acc / train_num
        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch

        with torch.no_grad():
            print("validation: ")
            for data_test in tqdm(valid_loader):
                test_images, test_labels = data_test
                outputs = model(test_images.to(device))  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == test_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, "restnet_{}.pth".format(args.model_name)))
            #         print('[epoch %d] train_loss: %.3f  test_acc: %.3f  train_acc: %.3f' %
            #               (epoch + 1, running_loss / step, val_accurate, train_accurate))
            # ******************************************************
            # for train_data_test in train_loader:
            #     train_images, train_labels = train_data_test
            #     train_outputs = model(train_images.to(device))  # eval model only have last output layer
            #     # loss = loss_function(outputs, test_labels)
            #     train_predict_y = torch.max(train_outputs, dim=1)[1]
            #     train_acc += (train_predict_y == train_labels.to(device)).sum().item()
            # train_accurate = train_acc / train_num
            #         if val_accurate > best_acc:
            #             best_acc = val_accurate
            # torch.save(net.state_dict(), save_path)
            # print(' train_acc: %.3f' %(train_accurate))
            print('[epoch %d] train_loss: %.3f  test_acc: %.3f  train_acc: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate, train_accurate))
            train_ac.append(train_accurate)
            vail_ac.append(val_accurate)
