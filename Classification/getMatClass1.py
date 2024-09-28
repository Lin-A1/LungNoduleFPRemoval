from xml.etree.ElementTree import Element, ElementTree
import csv
import pandas as pd
from Classification.getImg import load_itk_image, truncate_hu, normalazation
import os
import numpy as np
import scipy.io as io
from scipy import ndimage
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def beatau(e, level=0):
    if len(e) > 0:
        e.text = '\n' + '\t' * (level + 1)
        for child in e:
            beatau(child, level + 1)
        child.tail = child.tail[:-1]
    e.tail = '\n' + '\t' * level


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


def resample(image, spacing, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    # print("spacing:",spacing)
    # spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    # print("image.shape",image.shape)
    # print("new_shape",new_shape)
    image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing, real_resize_factor


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    # verts, faces = measure.marching_cubes_lewiner(p, threshold)
    print("step1")
    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold, spacing=(1., 1., 1.), allow_degenerate=True)
    print("step2")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    print("step3")
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    print("step4")
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    print("step5")
    ax.add_collection3d(mesh)
    print("step6")
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()
    print("step7")


if __name__ == "__main__":

    # 标签数据
    # fname = 'TestData/CSVFILES/annotations.csv'
    fname = 'TestData/CSVFILES/candidates.csv'
    mat_path = 'Classification/classfier/makeMat_/'
    data = pd.read_csv(fname)
    data = data[data['class'] == 1]
    data.reset_index(drop=True, inplace=True)
    # 遍历data标签  生成png和对应的xml
    nameLast = []  # 上一次的文件名
    namePre = []  # 这一次的文件名
    fileDir = []
    count = 1
    numNoudle = 0
    for i in range(len(data)):

        # 先获取.mhd 文件名
        # 测试3次读取输出，判断是否生成成功
        if i > len(data)-1:
            break
        print("{}   /total {} , count:{} ,num Noudles: {}  ".format(i, len(data), count, numNoudle))
        namePre = data['seriesuid'].loc[i]
        if namePre != nameLast:  # 如果数值不同 重新赋值并查找
            # print("different  ")
            # 获得路径
            fileDir.clear()  # 清空
            fileDir = search(path=r"data_/", name=namePre)
            getDir = None
            # 找到 .mhd文件
            for file in fileDir:
                if '.mhd' in file:
                    getDir = file
                    break

            if getDir is None:
                continue

        # 获取CT图像标注数据
        x_ano = data['coordX'].loc[i]
        y_ano = data['coordY'].loc[i]
        z_ano = data['coordZ'].loc[i]
        # r = TestData['diameter_mm'].loc[i]
        classMat = data['class'].loc[i]

        numpyimage, CT, isflip = load_itk_image(getDir)
        truncate_hu(numpyimage)  # 截取像素值
        image_array = normalazation(numpyimage)  # 数值归一化
        imgMat = image_array.transpose(1, 2, 0)  # transpose是将(z,x,y)的三维矩阵转为(x,y,z)的矩阵
        # 重采样为1mm*1mm*1mm
        pix_resampled, spacing, real_resize_factor = resample(imgMat, CT.ElementSpacing, [1, 1, 1])
        # plot_3d(pix_resampled ,100)
        # print("after resample")
        # print("spacing:",spacing)
        # print("real_resize_factor:",real_resize_factor)
        # 图像坐标计算
        """
        print("z_anp    :",z_ano)
        print("z_offset :",CT.z_offset)
        print("z_anp - CT.z_offset:",z_ano - CT.z_offset)
        """
        # 换算成重采样后的坐标 与长宽高
        x = np.round(worldToVoxelCoord(x_ano, CT.x_offset, CT.x_ElementSpacing) * real_resize_factor[0]).astype(int)
        y = np.round(worldToVoxelCoord(y_ano, CT.y_offset, CT.y_ElementSpacing) * real_resize_factor[1]).astype(int)
        z = np.round(worldToVoxelCoord(z_ano, CT.z_offset, CT.z_ElementSpacing) * real_resize_factor[2]).astype(int)
        """
        w = np.round(r/CT.x_ElementSpacing * real_resize_factor[0]).astype(int)
        h = np.round(r/CT.y_ElementSpacing * real_resize_factor[1]).astype(int)
        l = np.round(r/CT.z_ElementSpacing * real_resize_factor[2]).astype(int)
        """
        # print("x    :",x)
        # print("y    :",y)
        # print("z    :",z)
        # 转换成有标注的图片，判断生成的标注是否正确，并保存
        """
        x_min = x - np.round(w/2).astype(int)
        y_min = y - np.round(h/2).astype(int)
        x_max = x + np.round(w/2).astype(int)
        y_max = y + np.round(h/2).astype(int)
        """
        # 如果图像进行旋转了，那么坐标也要旋转
        ImageW = int(512 * real_resize_factor[0])
        ImageH = int(512 * real_resize_factor[1])
        # 如果图像进行旋转了，那么坐标也要旋转
        if isflip:
            x = ImageW - x
            y = ImageH - y

        # 切成40*40*24
        """
        mat = pix_resampled[x - int(w/2): x + int(w/2) + 1, 
                            y - int(h/2): y + int(h/2) + 1,
                            z - int(l/2): z + int(l/2) + 1]
        """
        mat = pix_resampled[y - 20: y + 20,
              x - 20: x + 20,
              z - 12: z + 12]

        # 对mat尺寸进行判断
        x_mat, y_mat, z_mat = mat.shape
        if x_mat * y_mat * z_mat < 38400:
            print("mat error:mat.shape:{} \n ID:{} \n x_ano:{},y_ano:{},z_ano:{},x:{},y:{},z:{}" \
                  .format(mat.shape, namePre, x_ano, y_ano, z_ano, x, y, z))
            continue  # 进行下一个循环
        if classMat == 1:
            numNoudle = numNoudle + 1  # 结节计数
            io.savemat(mat_path + '{:05d}_.mat'.format(count), {'TestData': mat, 'class': classMat})
            count = count + 1
        else:
            io.savemat(mat_path + '{:05d}_.mat'.format(count), {'TestData': mat, 'class': classMat})
            count = count + 1

        nameLast = namePre








