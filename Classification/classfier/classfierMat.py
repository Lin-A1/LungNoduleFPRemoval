import scipy.io as sio
import numpy as np
import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
import argparse
# 模型
from model import *
# 数据
from traindataset import *
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--data_size", type=int, default=10, help="size of each TestData dimension")
    parser.add_argument("--train_path", default="makeMat/train_data", help="the path of train TestData")
    parser.add_argument("--test_path", default="makeMat/test_data", help="the path of test TestData")
    parser.add_argument("--crop_size", default=[36, 36, 20], help="allow for multi-scale training")
    parser.add_argument("--thresh", default=0.5, help="i>thresh,i=1,else,i=0")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.makedirs() 方法用于递归创建目录。
    model = resnet3d(1, 1).to(device)

    # Get dataloader
    trian_dataset = MyTrainData(opt.train_path, opt.crop_size)
    test_dataset = MyTrainData(opt.test_path, opt.crop_size)
    """
    load_path = 'makeMat/0002.mat'
    load_data = sio.loadmat(load_path) 
    
    for key, value in load_data.items():
        print(key, ':', value)
    """
    train_data = torch.utils.data.DataLoader(
        trian_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
    )
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
    )
    # 优化器
    optimizer = torch.optim.Adam(model.parameters())
    # 损失函数
    criterion = nn.BCELoss()

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(opt.epochs):

        model.train()
        loss_sigma = 0.0
        correct = 0.0
        total = 0.0
        for batch_i, (imgs, targets) in enumerate(train_data):

            # print("batch_i:\n",batch_i)
            imgs = imgs.type(torch.FloatTensor)  # 转Float
            imgs = imgs.to(device)
            # print("imgs.shape:",imgs.shape)
            # 添加大小为1的维度
            imgs = torch.unsqueeze(imgs, 1)  # 在第1个维度上扩展
            targets = targets.type(torch.FloatTensor)  # 转float 这个会把gpu变成cpu
            targets = torch.squeeze(targets, 2)  # 删除第二个维度
            targets = targets.to(device)  # requires_grad = flase

            outputs = model(imgs)
            outputs = outputs.to(device)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_sigma = loss_sigma + loss

            total += targets.size(0)
            predict = torch.tensor(outputs)  # 复制
            predict[predict >= opt.thresh] = 1
            predict[predict < opt.thresh] = 0

            correct += (predict == targets).squeeze().sum().cpu().numpy()
            # print("correct: ",correct)
            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if batch_i % 10 == 0:
                loss_avg = loss_sigma / 10.0
                loss_sigma = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}" \
                      .format(epoch + 1, opt.epochs, batch_i + 1, len(train_data), loss_avg, correct / total))
        train_loss_list.append(loss_avg)
        train_acc_list.append(correct / total)

        # 模型在验证集上的表现情况
        if epoch % 1 == 0:
            loss_sigma = 0.0
            model.eval()
            correct = 0
            total = 0.0
            for batch_i, (imgs, targets) in enumerate(test_data):
                # forward
                imgs = imgs.type(torch.FloatTensor)  # 转Float
                imgs = imgs.to(device)
                imgs = torch.unsqueeze(imgs, 1)  # 在第1个维度上扩展

                targets = targets.type(torch.FloatTensor)  # 转float 这个会把gpu变成cpu
                targets = torch.squeeze(targets, 2)  # 删除第二个维度
                targets = targets.to(device)  # requires_grad = flase

                outputs = model(imgs)
                outputs.detach_()
                # 计算loss
                loss = criterion(outputs, targets)
                loss_sigma += loss.item()
                # 统计
                total += targets.size(0)
                predict = torch.tensor(outputs)  # 复制
                predict[predict >= opt.thresh] = 1
                predict[predict < opt.thresh] = 0
                # print("predict: ",predict)
                # print("predict.shape: ",predict.shape)
                # print("targets: ",targets)
                # print("targets.shape: ",targets.shape)

                correct += (predict == targets).squeeze().sum().cpu().numpy()
                # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
                if batch_i % 10 == 0:
                    loss_avg = loss_sigma / 10.0
                    loss_sigma = 0.0
                    print("test: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}" \
                          .format(epoch + 1, opt.epochs, batch_i + 1, len(test_data), loss_avg, correct / total))
        test_loss_list.append(loss_avg)
        test_acc_list.append(correct / total)

        # 每个epcoch 保存一次
        torch.save(model, 'model/net_' + str(epoch) + '.pkl')  # 保存整个神经网络的结构和模型参数
        torch.save(model.state_dict(), 'model/net_ALL_' + str(epoch) + '.pkl')  # 只保存神经网络的模型参数

    train_loss_list = [x.item() for x in train_loss_list]
    train_loss_list[0] = 10*train_loss_list[0]
    test_loss_list[0] = 10*test_loss_list[0]

    # 创建一个2行2列的子图布局
    plt.figure(figsize=(10, 8))

    # 第一个子图：训练损失
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_list, label='Train Loss', color='blue')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 第二个子图：测试损失
    plt.subplot(2, 2, 2)
    plt.plot(test_loss_list, label='Test Loss', color='red')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 第三个子图：训练准确率
    plt.subplot(2, 2, 3)
    plt.plot(train_acc_list, label='Train Accuracy', color='green')
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 第四个子图：测试准确率
    plt.subplot(2, 2, 4)
    plt.plot(test_acc_list, label='Test Accuracy', color='orange')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 调整子图布局
    plt.tight_layout()

    # 保存图表
    plt.savefig('training_results.png')

