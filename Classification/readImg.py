import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_mhd_image(filename):
    """加载 .mhd 文件并返回 numpy 数组格式的图像数据"""
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    return ct_scan


def plot_slice(ct_scan, slice_number=0):
    """可视化指定切片号的 CT 切片"""
    plt.imshow(ct_scan[slice_number], cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_first_slice(mhd_file_path):
    """加载 .mhd 文件并可视化第一个切片"""
    ct_scan = load_mhd_image(mhd_file_path)
    plot_slice(ct_scan)


if __name__ == "__main__":
    mhd_file_path = '../data/1.3.6.1.4.1.14519.5.2.1.6279.6001.232011770495640253949434620907.mhd'  # 替换为您的 .mhd 文件路径
    visualize_first_slice(mhd_file_path)
