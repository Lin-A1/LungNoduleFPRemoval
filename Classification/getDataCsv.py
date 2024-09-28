import csv
import json
import os
from xml.etree.ElementTree import Element, ElementTree, tostring

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from getImg import load_itk_image, truncate_hu, normalazation, getLungMask


def csvtoxml(fname):
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        root = Element('Daaa')
        print('root', len(root))
        for row in reader:
            erow = Element('Row')
            root.append(erow)
            for tag, text in zip(header, row):
                e = Element(tag)
                e.text = text
                erow.append(e)
    beatau(root)
    return ElementTree(root)


def beatau(e, level=0):
    if len(e) > 0:
        e.text = '\n' + '\t' * (level + 1)
        for child in e:
            beatau(child, level + 1)
        child.tail = child.tail[:-1]
    e.tail = '\n' + '\t' * level


# 世界坐标转换到图像中的坐标
def worldToVoxelCoord(worldCoord, offset, EleSpacing):
    stretchedVoxelCoord = np.absolute(worldCoord - offset)
    voxelCoord = stretchedVoxelCoord / EleSpacing

    return voxelCoord


# 图像上的坐标转换为世界坐标：
def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin

    return worldCoord


def ToXml(name, x, y, w, h):
    root = Element('annotation')  # 根节点
    erow1 = Element('folder')  # 节点1
    erow1.text = "VOC"

    erow2 = Element('filename')  # 节点2
    erow2.text = str(name)

    erow3 = Element('size')  # 节点3
    erow31 = Element('width')
    erow31.text = "512"
    erow32 = Element('height')
    erow32.text = "512"
    erow33 = Element('depth')
    erow33.text = "3"
    erow3.append(erow31)
    erow3.append(erow32)
    erow3.append(erow33)

    erow4 = Element('object')
    erow41 = Element('name')
    erow41.text = 'nodule'
    erow42 = Element('bndbox')
    erow4.append(erow41)
    erow4.append(erow42)
    erow421 = Element('xmin')
    erow421.text = str(x - np.round(w / 2).astype(int))
    erow422 = Element('ymin')
    erow422.text = str(y - np.round(h / 2).astype(int))
    erow423 = Element('xmax')
    erow423.text = str(x + np.round(w / 2).astype(int))
    erow424 = Element('ymax')
    erow424.text = str(y + np.round(h / 2).astype(int))
    erow42.append(erow421)
    erow42.append(erow422)
    erow42.append(erow423)
    erow42.append(erow424)

    root.append(erow1)
    root.append(erow2)
    root.append(erow3)
    root.append(erow4)
    beatau(root)

    return ElementTree(root)


# 查询文件 返回路径   -1表示无
def search(path=".", name="", fileDir=[]):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            search(item_path, name)
        elif os.path.isfile(item_path):
            if name in item:
                fileDir.append(item_path)
                # print("fileDir:",fileDir)

    return fileDir


if __name__ == "__main__":

    # 标签数据
    fname = 'TestData/CSVFILES/annotations.csv'
    data = pd.read_csv(fname)
    # 遍历data标签  生成png和对应的xml
    nameLast = []  # 上一次的文件名
    namePre = []  # 这一次的文件名
    fileDir = []
    count = 0
    for i in range(len(data)):
        # 先获取.mhd 文件名
        # 测试3次读取输出，判断是否生成成功
        # if i > 10:
        # break
        print("{}   /total {}   ".format(i, len(data)))
        namePre = data['seriesuid'].loc[i]
        if namePre != nameLast:  # 如果数值不同 重新赋值并查找
            # print("different  ")
            # 获得路径
            fileDir.clear()  # 清空
            fileDir = search(path=r"../data/", name=namePre)
            # 找到 .mhd文件
            for file in fileDir:
                if '.mhd' in file:
                    getDir = file
                    break

        # 获取CT图像标注数据
        x_ano = data['coordX'].loc[i]
        y_ano = data['coordY'].loc[i]
        z_ano = data['coordZ'].loc[i]
        r = data['diameter_mm'].loc[i]

        numpyimage, CT, isflip = load_itk_image(getDir)
        truncate_hu(numpyimage)  # 截取像素值
        image_array = normalazation(numpyimage)  # 数值归一化
        # 图像坐标计算
        """
        print("z_anp    :",z_ano)
        print("z_offset :",CT.z_offset)
        print("z_anp - CT.z_offset:",z_ano - CT.z_offset)
        """
        x = np.round(worldToVoxelCoord(x_ano, CT.x_offset, CT.x_ElementSpacing)).astype(int)
        y = np.round(worldToVoxelCoord(y_ano, CT.y_offset, CT.y_ElementSpacing)).astype(int)
        z = np.round(worldToVoxelCoord(z_ano, CT.z_offset, CT.z_ElementSpacing)).astype(int)
        w = np.round(r / CT.x_ElementSpacing).astype(int)
        h = np.round(r / CT.y_ElementSpacing).astype(int)
        # print("x    :",x)
        # print("y    :",y)
        # print("z    :",z)
        """
        #切割图像 切割相邻的3张图片
        imgLabel1 = image_array.transpose(1,2,0)[:,:,z - 1] #transpose是将(z,x,y)的三维矩阵转为(x,y,z)的矩阵
        imgLabel2 = image_array.transpose(1,2,0)[:,:,z] #transpose是将(z,x,y)的三维矩阵转为(x,y,z)的矩阵
        imgLabel3 = image_array.transpose(1,2,0)[:,:,z + 1] #transpose是将(z,x,y)的三维矩阵转为(x,y,z)的矩阵
        #转换成RGB类型并保存
        #现转换成肺实质分割之后的图片
        img, mask = getLungMask(imgLabel1)
        im1 = Image.fromarray(img)
        im1 = im1.convert("RGB")      
        im1.save('VOCdevkit/VOC2019/JPEGImages/{:04d}.png'.format(count+1))
        count = count + 1
        img, mask = getLungMask(imgLabel2)
        im2 = Image.fromarray(img)
        im2 = im2.convert("RGB")      
        im2.save('VOCdevkit/VOC2019/JPEGImages/{:04d}.png'.format(count+1))
        count = count + 1
        img, mask = getLungMask(imgLabel3)
        im3 = Image.fromarray(img)
        im3 = im3.convert("RGB")      
        im3.save('VOCdevkit/VOC2019/JPEGImages/{:04d}.png'.format(count+1))
        count = count + 1
        """
        count = count + 3
        # 转换成有标注的图片，判断生成的标注是否正确，并保存
        x_min = x - np.round(w / 2).astype(int)
        y_min = y - np.round(h / 2).astype(int)
        x_max = x + np.round(w / 2).astype(int)
        y_max = y + np.round(h / 2).astype(int)
        # 如果图像进行旋转了，那么坐标也要旋转
        if isflip:
            x_min = 512 - x_min
            y_min = 512 - y_min
            x_max = 512 - x_max
            y_max = 512 - y_max
            x = 512 - x
            y = 512 - y
        """
        #创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(im1)
        #线条只有应该像素点宽，多画几次变粗        
        draw.polygon([(x_min-1,y_min-1),(x_min-1,y_max+1),(x_max+1,y_max+1),(x_max+1,y_min-1)], outline=(255,0,0))
        draw.polygon([(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)], outline=(255,0,0))
        draw.polygon([(x_min+1,y_min+1),(x_min+1,y_max-1),(x_max-1,y_max-1),(x_max-1,y_min+1)], outline=(255,0,0))
        im1.save('VOCdevkit/VOC2019/LabelImages/{:04d}.png'.format(count - 2))

        #创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(im2)
        #线条只有应该像素点宽，多画几次变粗        
        draw.polygon([(x_min-1,y_min-1),(x_min-1,y_max+1),(x_max+1,y_max+1),(x_max+1,y_min-1)], outline=(255,0,0))
        draw.polygon([(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)], outline=(255,0,0))
        draw.polygon([(x_min+1,y_min+1),(x_min+1,y_max-1),(x_max-1,y_max-1),(x_max-1,y_min+1)], outline=(255,0,0))
        im2.save('VOCdevkit/VOC2019/LabelImages/{:04d}.png'.format(count - 1))

        #创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(im3)
        #线条只有应该像素点宽，多画几次变粗        
        draw.polygon([(x_min-1,y_min-1),(x_min-1,y_max+1),(x_max+1,y_max+1),(x_max+1,y_min-1)], outline=(255,0,0))
        draw.polygon([(x_min,y_min),(x_min,y_max),(x_max,y_max),(x_max,y_min)], outline=(255,0,0))
        draw.polygon([(x_min+1,y_min+1),(x_min+1,y_max-1),(x_max-1,y_max-1),(x_max-1,y_min+1)], outline=(255,0,0))
        im3.save('VOCdevkit/VOC2019/LabelImages/{:04d}.png'.format(count))
        """
        # 保存XML标注
        xmlLabel = ToXml('{:04d}.png'.format(count - 2), x, y, w, h)
        xmlLabel.write('./VOCdevkit/VOC2019/Annotations/{:04}.xml'.format(count - 2))
        xmlLabel = ToXml('{:04d}.png'.format(count - 1), x, y, w, h)
        xmlLabel.write('./VOCdevkit/VOC2019/Annotations/{:04}.xml'.format(count - 1))
        xmlLabel = ToXml('{:04d}.png'.format(count), x, y, w, h)
        xmlLabel.write('./VOCdevkit/VOC2019/Annotations/{:04}.xml'.format(count))
        nameLast = namePre
