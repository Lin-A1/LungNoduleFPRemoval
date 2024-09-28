import matplotlib

matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
from model import *
from traindataset import MyTrainData


def predict(model, data_loader, device):
    model.eval()
    all_scores = []

    with torch.no_grad():
        for batch_i, (imgs, _) in enumerate(data_loader):  # 不再需要真实标签
            imgs = imgs.type(torch.FloatTensor).to(device)
            imgs = torch.unsqueeze(imgs, 1)

            outputs = model(imgs)
            scores = outputs.cpu().numpy()  # 仅收集预测分数，而不是二进制预测
            all_scores.append(scores)

    all_scores = np.concatenate(all_scores)

    return all_scores


def visualize_slices_with_predictions(images, predictions):
    # 创建一个8x8的子图
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))

    for i in range(64):
        ax = axes[i // 8, i % 8]
        ax.imshow(images[i, 0, :, :], cmap='gray')  # 这里假设输入图像是单通道的
        ax.axis('off')
        ax.set_title(f"Pred: {predictions[i][0]:.2f}", fontsize=8)  # 修改此行

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设置设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet3d(1, 1).to(device)
    model.load_state_dict(torch.load("model/net_ALL_78.pkl", map_location=device))
    model.eval()

    # 加载测试数据集
    test_dataset = MyTrainData("makeMat/test_data", [36, 36, 20])
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,  # 修改批量大小为64
        shuffle=True,  # 随机抽取图像
        num_workers=0,
        pin_memory=True,
    )

    # 进行预测
    all_scores = predict(model, test_data, device)

    # 获取测试数据的切片
    sample_batch, _ = next(iter(test_data))

    # 可视化切片和预测结果
    visualize_slices_with_predictions(sample_batch.cpu().numpy(), all_scores)
