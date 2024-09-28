from tool.getMat import *

from tool.readpred import *


def readPred(model_path, data_path, batch_size=64, num_workers=0):
    # 设置设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet3d(1, 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载测试数据集
    test_dataset = MyTrainData(data_path, [36, 36, 20])
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 进行预测
    all_scores = predict(model, test_data, device)

    # 获取测试数据的切片
    sample_batch, _ = next(iter(test_data))

    # 可视化切片和预测结果
    visualize_slices_with_predictions(sample_batch.cpu().numpy(), all_scores)


def getMat(fname, mat_path,ct_path):
    data = pd.read_csv(fname)
    nameLast = []
    namePre = []
    fileDir = []
    count = 1
    numNoudle = 0
    for i in range(len(data)):
        if i > 551060:
            break
        print("{}   /total {} , count:{} ,num Noudles: {}  ".format(i, len(data), count, numNoudle))
        namePre = data['seriesuid'].loc[i]
        if namePre != nameLast:
            fileDir.clear()
            fileDir = search(path=ct_path, name=namePre)
            getDir = None
            for file in fileDir:
                if '.mhd' in file:
                    getDir = file
                    break

            if getDir is None:
                continue

        x_ano = data['coordX'].loc[i]
        y_ano = data['coordY'].loc[i]
        z_ano = data['coordZ'].loc[i]
        classMat = data['class'].loc[i]

        numpyimage, CT, isflip = load_itk_image(getDir)
        truncate_hu(numpyimage)
        image_array = normalazation(numpyimage)
        imgMat = image_array.transpose(1, 2, 0)
        pix_resampled, spacing, real_resize_factor = resample(imgMat, CT.ElementSpacing, [1, 1, 1])

        x = np.round(worldToVoxelCoord(x_ano, CT.x_offset, CT.x_ElementSpacing) * real_resize_factor[0]).astype(int)
        y = np.round(worldToVoxelCoord(y_ano, CT.y_offset, CT.y_ElementSpacing) * real_resize_factor[1]).astype(int)
        z = np.round(worldToVoxelCoord(z_ano, CT.z_offset, CT.z_ElementSpacing) * real_resize_factor[2]).astype(int)

        ImageW = int(512 * real_resize_factor[0])
        ImageH = int(512 * real_resize_factor[1])
        if isflip:
            x = ImageW - x
            y = ImageH - y

        mat = pix_resampled[y - 20: y + 20,
              x - 20: x + 20,
              z - 12: z + 12]

        x_mat, y_mat, z_mat = mat.shape
        if x_mat * y_mat * z_mat < 38400:
            print("mat error:mat.shape:{} \n ID:{} \n x_ano:{},y_ano:{},z_ano:{},x:{},y:{},z:{}" \
                  .format(mat.shape, namePre, x_ano, y_ano, z_ano, x, y, z))
            continue
        if classMat == 1:
            numNoudle = numNoudle + 1
            io.savemat(mat_path + '{:05d}.mat'.format(count), {'TestData': mat, 'class': classMat})
            count = count + 1
        else:
            io.savemat(mat_path + '{:05d}.mat'.format(count), {'TestData': mat, 'class': classMat})
            count = count + 1

        nameLast = namePre


if __name__ == "__main__":
    fname = 'TestData/CSVFILES/candidates.csv'
    ct_path = 'TestData'
    mat_path = 'data/makeMat_/'
    getMat(fname, mat_path)

    model_path = "model/net_ALL_78.pkl"
    data_path = "data/makeMat_"
    readPred(model_path, data_path)
