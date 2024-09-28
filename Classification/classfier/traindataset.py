# encoding:utf-8
import torch.utils.data as data
import torch

import os
import os.path
import glob
import random
from torchvision import transforms
import scipy.io as sio

import torchsnooper


def getAllDataPath(dataPath):  # 获取路径下所有文件路径
    pathAll = []
    for root, dirs, files in os.walk(dataPath):
        path = [os.path.join(root, name) for name in files]
        # print(path)
        pathAll.extend(path)
    return pathAll


# 数据增强
# @torchsnooper.snoop()
def dataAugmentation(dataMat, cropW, cropH, cropD, classID):
    # 随机裁剪
    dataMat = dataMat.transpose(2, 0, 1)  # 转置 从x,y,z到z,x,y
    D, W, H = dataMat.shape
    # 转换成张量
    dataMat = torch.from_numpy(dataMat)

    # print("dataMat.shape:",dataMat.shape)
    """
    print("W: ",W)
    print("H: ",H)
    print("D: ",D)
    print("cropW: ",cropW)
    print("cropH: ",cropH)
    print("cropD: ",cropD)
    """
    randX = random.randint(0, W - cropW)
    randY = random.randint(0, H - cropH)
    randZ = random.randint(0, D - cropD)
    """
    print("randX: ",randX)
    print("randY: ",randY)
    print("randZ: ",randZ)
    """
    dataMat = dataMat[randZ: randZ + cropD, randX: randX + cropW, randY: randY + cropH]
    # 随机翻转
    if classID == 1:  # 只对肺结节进行翻转扩充
        randDim = random.randint(0, 7)
        dims = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
        if randDim < 7:
            dataMat = torch.flip(dataMat, dims[randDim])

    return dataMat


# 自定義dataset的框架
class MyTrainData(data.Dataset):  # 需要繼承data.Dataset
    def __init__(self, dataPath, cropSize, transform=None):  # 初始化文件路進或文件名
        self.dataPath = getAllDataPath(dataPath)
        self.cropW = cropSize[0]
        self.cropH = cropSize[1]
        self.cropD = cropSize[2]

    def __getitem__(self, idx):
        dataMatPath = self.dataPath[idx]
        # print("dataMatPath: ",dataMatPath)
        # 加载.mat文件
        load_data = sio.loadmat(dataMatPath)

        # 获取Mat值和Class类别
        dataMat = load_data["data"]
        classMat = load_data["class"]

        # 归一化到 0~1之间
        dataMat = dataMat / 1.0

        dataMat = dataAugmentation(dataMat, self.cropW, self.cropH, self.cropD, classMat)  # 裁剪 翻转对称

        return dataMat, classMat  # 返回Mat的数据（numpy）和类别（0、1）

    def __len__(self):
        return len(self.dataPath)
