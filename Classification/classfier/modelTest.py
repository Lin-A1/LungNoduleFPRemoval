import matplotlib
from matplotlib import pyplot as plt
import argparse
import warnings

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, \
    precision_recall_curve, classification_report, confusion_matrix

from model import *
from traindataset import MyTrainData

matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")


def predict_and_evaluate(model, data_loader, device):
    model.eval()
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for batch_i, (imgs, targets) in enumerate(data_loader):
            imgs = imgs.type(torch.FloatTensor).to(device)
            imgs = torch.unsqueeze(imgs, 1)
            targets = targets.type(torch.FloatTensor).to(device)
            targets = torch.squeeze(targets, 2)

            outputs = model(imgs)
            scores = outputs.cpu().numpy()  # Collect predicted scores instead of binary predictions
            all_scores.append(scores)
            all_targets.append(targets.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)

    return all_scores, all_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/lin/work/code/DeepLearnling/Image "
                                                "Classification/LungNoduleFPRemoval/Classification/classfier/model"
                                                "/net_ALL_78.pkl", help="训练好的模型的路径")
    parser.add_argument("--test_path", default="makeMat/test_data", help="测试数据的路径")
    parser.add_argument("--batch_size", type=int, default=64, help="每批图像的大小")
    parser.add_argument("--threshold", type=float, default=0.5, help="用于二分类的决策阈值")
    parser.add_argument("--crop_size", default=[36, 36, 20], help="allow for multi-scale training")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet3d(1, 1).to(device)
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model.eval()

    test_dataset = MyTrainData(opt.test_path, opt.crop_size)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    all_scores, all_targets = predict_and_evaluate(model, test_data, device)

    # 已有的指标计算
    auc = roc_auc_score(all_targets, all_scores)
    accuracy = accuracy_score(all_targets, (all_scores >= opt.threshold).astype(int))
    sensitivity = recall_score(all_targets, (all_scores >= opt.threshold).astype(int))
    ppv = precision_score(all_targets, (all_scores >= opt.threshold).astype(int))

    # 计算混淆矩阵
    predictions = (all_scores >= opt.threshold).astype(int)
    cm = confusion_matrix(all_targets, predictions)
    TN, FP, FN, TP = cm.ravel()

    # 计算特异性
    specificity = TN / (TN + FP)

    # 计算NPV
    npv = TN / (TN + FN)

    # 计算PLR和NLR
    plr = sensitivity / (1 - specificity)
    nlr = (1 - sensitivity) / specificity

    print(classification_report(all_targets, (all_scores >= opt.threshold).astype(int)))

    precision, recall, _ = precision_recall_curve(all_targets, all_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    fpr, tpr, _ = roc_curve(all_targets, all_scores)

    plt.figure(figsize=(15, 5))  # 创建一个大图，包含3个子图

    # ROC 曲线子图
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.5f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 精确-召回曲线子图
    plt.subplot(1, 3, 2)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    # F1-Score 曲线子图
    plt.subplot(1, 3, 3)
    thresholds = np.linspace(0, 1, num=len(precision))  # Generate thresholds
    plt.plot(thresholds, f1_scores, 'b-')  # Keep f1_scores as they are
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Curve')

    plt.tight_layout()  # 调整子图布局，避免重叠
    plt.savefig('test_results/evaluation_plots.png')
    plt.show()

    print("Area under the curve (曲线下面积):", auc)
    print("Accuracy (准确度):", accuracy)
    print("Sensitivity (敏感度):", sensitivity)
    print("Specificity (特异性):", specificity)
    print("Positive predictive value (正向预测值):", ppv)
    print("Negative predictive value (负向预测值):", npv)
    print("Positive likelihood ratio (正似然比):", plr)
    print("Negative likelihood ratio (负似然比):", nlr)

    # 将评估指标存储为字典
    evaluation_results = {
        "指标": ["Area under the curve (曲线下面积)", "Accuracy (准确度)", "Sensitivity (敏感度)", "Specificity (特异性)",
                "Positive predictive value (正向预测值)", "Negative predictive value (负向预测值)",
                "Positive likelihood ratio (正似然比)", "Negative likelihood ratio (负似然比)"],
        "分数": [auc, accuracy, sensitivity, specificity, ppv, npv, plr, nlr]
    }

    # 创建 DataFrame 对象
    df = pd.DataFrame(evaluation_results)
    df.to_excel('test_results/results.xlsx')
