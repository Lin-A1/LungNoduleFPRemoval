# LungNoduleFPRemoval

肺结节假阳性剔除模型

本次项目旨在完成肺结节假阳性的剔除，根本上属于三维图像的分类任务

- 数据集
```
luna16

分享下载链接：https://blog.csdn.net/weixin_48335963/article/details/120214931
```

- 下载环境

```
pip install -r requirements.txt
```

- 数据集位置

```
data
```

- 数据预处理

```
python getMat.py
python classfier/moveMat.py
```

- 模型训练

```
python classfier/classfierMat.py
```
<img src="./Classification/classfier/training_results.png" alt="training_results" style="zoom:50%;" />

- 模型测试

```
python classfier/modelTest.py
```

<img src="./Classification/classfier/test_results/evaluation_plots.png" alt="evaluation_plots" style="zoom: 50%;" />

| 指标                      | 值          |
| ------------------------- | ----------- |
| Area under the curve      | 0.990087069 |
| Accuracy                  | 0.964726631 |
| Sensitivity               | 0.945762712 |
| Specificity               | 0.971394517 |
| Positive predictive value | 0.920792079 |
| Negative predictive value | 0.980746089 |
| Positive likelihood ratio | 33.06228814 |
| Negative likelihood ratio | 0.05583446  |

- qt页面

```
python run ./Classification/qt/checkQt.py
```

 ps:若出现找不到文件，可以查看`.gitignore`由于`github`限制文件传输大小，我屏蔽了部分数据集、模型内容，现在保留的内容只有预处理过的数据，模型需要自己跑