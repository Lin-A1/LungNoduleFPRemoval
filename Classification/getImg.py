import SimpleITK as sitk
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from sklearn.cluster import KMeans

"""
import scipy.misc
scipy.misc.toimage(image_array).save('outfile.jpg')
"""

import cv2

"""
x= (x_ano-x_offset)/x_ElementSpacing 
y= (y_ano-y_offset)/y_ElementSpacing 
z= (z_ano-z_offset)/z_ElementSpacing 


其中，x是实际对应三维矩阵中的坐标。
x_ano是肺结节在annotation.csv中的坐标.
x_offset是质心坐标.
x_ElementSpacing是在x轴方向上的步长。相当于每一个像素对应现实世界中的长度。
"""


# 定义CT图像类保存 中心点
class CTImage(object):
    """docstring for Hotel"""

    def __init__(self, x_offset, y_offset, z_offset, x_ElementSpacing, y_ElementSpacing, z_ElementSpacing):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.x_ElementSpacing = x_ElementSpacing
        self.y_ElementSpacing = y_ElementSpacing
        self.z_ElementSpacing = z_ElementSpacing
        self.ElementSpacing = np.array([x_ElementSpacing, y_ElementSpacing, z_ElementSpacing])


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        offset = [k for k in contents if k.startswith('Offset')][0]
        EleSpacing = [k for k in contents if k.startswith('ElementSpacing')][0]

        # 把值进行分割
        offArr = np.array(offset.split(' = ')[1].split(' ')).astype('float')
        eleArr = np.array(EleSpacing.split(' = ')[1].split(' ')).astype('float')
        CT = CTImage(offArr[0], offArr[1], offArr[2], eleArr[0], eleArr[1], eleArr[2])
        transform = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transform = np.round(transform)  # round() 方法返回浮点数x的四舍五入值
        if np.any(transform != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):  # 判断是否相等
            isflip = True
        else:
            isflip = False
    itkimage = sitk.ReadImage(filename)
    numpyimage = sitk.GetArrayFromImage(itkimage)
    if isflip:
        numpyimage = numpyimage[:, ::-1, ::-1]  # ::-1 倒序
    return numpyimage, CT, isflip


def truncate_hu(image_array):
    image_array[image_array > 400] = 400
    image_array[image_array < -1000] = -1000


def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    # 归一化
    image_array = (image_array - min) / (max - min) * 255
    # image_array = image_array.astype(int)#整型
    image_array = np.round(image_array)
    return image_array


# 找到较大连通域区域的索引
def findMaxRegion(img):
    # 计算连通域
    img = img.astype(np.uint8)
    num, labels = cv2.connectedComponents(img, connectivity=4)
    # 找到连通域的最大值
    getLabel = [0]
    for i in range(1, num):
        getLabel.append(np.sum(labels == i))
    # 求得最大连通域的索引
    maxNum = 0
    for i in getLabel:
        if i > maxNum:
            maxNum = i
    getIndex = []
    for i in range(num):
        if getLabel[i] > maxNum / 2:
            getIndex.append(i)
    maskLabelInedx = [np.where(labels == x) for x in getIndex]
    # 生成mask
    maskLabel = np.zeros(labels.shape, dtype=int)
    for i in range(len(maskLabelInedx)):
        maskLabel[maskLabelInedx[i]] = 1
    return maskLabel


def fill_color_demo(image):
    copyImage = image.copy()
    h, w = copyImage.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)  # 这里必须为 h+2,w+2
    cv2.floodFill(copyImage, mask, (20, 20), (0, 125, 125), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    cv2.imshow("fill_color_demo", copyImage)


def getLungMask(imNoudle):
    img = imNoudle
    # 1. 标准化数据
    # Standardize the pixel values
    mean = np.mean(imNoudle)
    std = np.std(imNoudle)
    imNoudle = imNoudle - mean
    imNoudle = imNoudle / std
    # 查看数值分布
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    ax1.imshow(imNoudle, cmap='gray')
    plt.hist(imNoudle.flatten(), bins=200)
    plt.show()
    # cv2.imshow("1",imNoudle)
    # 找出肺部附近的平均像素值，对洗掉的图像进行重新正规化
    middle = imNoudle[100:400, 100:400]
    mean = np.mean(middle)
    # 使用Kmeans分离前景(不透明组织)和背景(透明组织，即肺)
    # 这样做只在图像的中心，尽量避免非组织部分的图像
    # np.prod()函数用来计算所有元素的乘积
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(imNoudle < threshold, 1.0, 0.0)  # threshold the image
    # cv2.imshow("thresh_img",thresh_img)
    image_array = thresh_img
    plt.imshow(image_array, cmap='gray')
    plt.show()
    # 最初的侵蚀对去除某些区域的颗粒状很有帮助，然后使用大的膨胀使肺区域吞噬血管
    erosion = cv2.erode(thresh_img, np.ones((4, 4), np.uint8))  # 腐蚀
    dilation = cv2.dilate(erosion, np.ones((10, 10), np.uint8))  # 膨胀
    # dilation = dilation * np.ones(dilation.shape, dtype = np.uint8)
    # cv2.imshow("after change",dilation)
    # eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    # dilation = morphology.dilation(eroded,np.ones([10,10]))
    # 在skimage包中，使用measure子模块下的label函数即可实现连通区域标记。
    # 参数input表示需要处理的二值图像，connectivity表示判定连通的模式（1代表4连通，2代表8连通），
    # 输出labels为一个从0开始的标记数组。
    labels = measure.label(dilation)

    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    ax[0, 0].imshow(thresh_img, cmap='gray')
    ax[0, 1].imshow(erosion, cmap='gray')
    ax[1, 0].imshow(dilation, cmap='gray')
    ax[1, 1].imshow(labels)  # 标注mask区域切片图
    plt.show()

    # np.unique 该函数是去除数组中的重复数字，并进行排序之后输出
    label_vals = np.unique(labels)
    # measure.regionprops筛选连通区域
    regions = measure.regionprops(labels)
    good_labels = []
    """
    area	int	区域内像素点总数
    bbox	tuple	边界外接框(min_row, min_col, max_row, max_col)
    centroid	array　　	质心坐标
    convex_area	int	凸包内像素点总数
    convex_image	ndarray	和边界外接框同大小的凸包　　
    coords	ndarray	区域内像素点坐标
    Eccentricity 	float	离心率
    label	int	区域标记
    """
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0
    # 这里的mask是肺用的，不是肺结节用的，在只剩下肺之后，
    # 我们再做一次大的膨胀来填充和取出肺部mask
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = mask.astype(np.uint8)
    # mask = cv2.erode(mask, np.ones((10, 10)))#腐蚀会造成一些空洞
    mask = cv2.dilate(mask, np.ones((10, 10)))  # 膨胀填充空洞
    # mask = morphology.erosion(mask,np.ones([10,10]))#腐蚀 滤除一些边缘

    maskLabel = findMaxRegion(mask)

    fig, ax = plt.subplots(2, 2, figsize=[10, 10])
    ax[0, 0].imshow(img)  # CT切片图
    ax[0, 1].imshow(img, cmap='gray')  # CT切片灰度图
    ax[1, 0].imshow(maskLabel, cmap='gray')  # 标注mask，标注区域为1，其他为0
    ax[1, 1].imshow(img * maskLabel, cmap='gray')  # 标注mask区域切片图
    plt.show()

    # cv2.imshow("after mask imNoudle",(img * mask).astype(np.uint8))
    thresh_img_mask = np.ones(imNoudle.shape, dtype=np.uint8) * mask * 255
    thresh_img_mask = thresh_img_mask.astype(np.uint8)
    fill_color_demo(thresh_img_mask)
    # cv2.imshow("1thresh_img_mask",thresh_img_mask)
    """
    """
    maskLabel = findMaxRegion(mask)
    img = img * maskLabel
    img = img.astype(np.uint8)

    thresh_img_mask = np.ones(imNoudle.shape, dtype=np.uint8) * maskLabel * 255
    thresh_img_mask = thresh_img_mask.astype(np.uint8)
    imageio.imwrite('thresh_img_mask2.png', thresh_img_mask)
    # cv2.imshow("2thresh_img_mask",thresh_img_mask)
    # cv2.imshow("end",img)
    # cv2.waitKey(0)
    """
    """
    return img, maskLabel


if __name__ == "__main__":
    case_path = 'data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.126264578931778258890371755354.mhd'  # 测试的为0子集代码，在当地没保存其数据
    numpyimage, _, _ = load_itk_image(case_path)
    print("numpyimage:\n", type(numpyimage))
    print("numpyimage.shape: ", numpyimage.shape)
    imNoudle = numpyimage.transpose(1, 2, 0)[:, :, 170]

    truncate_hu(numpyimage)
    image_array = normalazation(numpyimage)
    imNoudle = image_array.transpose(1, 2, 0)[:, :, 600]
    img = imNoudle
    img2 = imNoudle
    imNoudle = imNoudle.astype(np.uint8)
    cv2.imshow("begin imNoudle", imNoudle)
    cv2.imwrite("imNoudle.png", imNoudle)
    print("image_array.shape", image_array.shape)
    img, mask = getLungMask(img)
    cv2.imshow("11", img)
    cv2.waitKey(0)
