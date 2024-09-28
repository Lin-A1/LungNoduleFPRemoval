import wx

from tool.readImg import *
from tool.traindataset import *
from tools import *


class LungNoduleDetectionApp(wx.Frame):
    def __init__(self, parent, title):
        super(LungNoduleDetectionApp, self).__init__(parent, title=title, size=(600, 250))

        self.panel = wx.Panel(self)

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        file_sizer = wx.GridSizer(3, 3, 10, 10)

        # 创建上传文件的按钮和文本框
        self.csv_button = wx.Button(self.panel, label="上传CSV文件")
        self.csv_text = wx.TextCtrl(self.panel, size=(300, -1))

        self.mhd_button = wx.Button(self.panel, label="上传MHD文件")
        self.mhd_text = wx.TextCtrl(self.panel, size=(300, -1))

        self.raw_button = wx.Button(self.panel, label="上传RAW文件")
        self.raw_text = wx.TextCtrl(self.panel, size=(300, -1))

        file_sizer.Add(self.csv_button, 0, wx.EXPAND)
        file_sizer.Add(self.csv_text, 0, wx.EXPAND)
        file_sizer.AddSpacer(0)

        file_sizer.Add(self.mhd_button, 0, wx.EXPAND)
        file_sizer.Add(self.mhd_text, 0, wx.EXPAND)
        file_sizer.AddSpacer(0)

        file_sizer.Add(self.raw_button, 0, wx.EXPAND)
        file_sizer.Add(self.raw_text, 0, wx.EXPAND)
        file_sizer.AddSpacer(0)

        main_sizer.Add(file_sizer, 0, wx.ALL | wx.EXPAND, 10)

        # # 创建显示3D图像的窗口
        # self.canvas = wx.Window(self.panel, size=(780, 400))
        # self.canvas.SetBackgroundColour(wx.Colour(255, 255, 255))
        # main_sizer.Add(self.canvas, 1, wx.ALL | wx.EXPAND, 10)

        # 创建切割和预测按钮
        self.segment_button = wx.Button(self.panel, label="进行肺结节切割并预测")
        main_sizer.Add(self.segment_button, 0, wx.ALL | wx.CENTER, 10)

        # 创建可视化切片按钮
        self.visualize_button = wx.Button(self.panel, label="肺部ct切片可视化")
        main_sizer.Add(self.visualize_button, 0, wx.ALL | wx.CENTER, 10)

        self.panel.SetSizer(main_sizer)

        # 绑定上传文件按钮的事件
        self.csv_button.Bind(wx.EVT_BUTTON, self.on_upload_csv)
        self.mhd_button.Bind(wx.EVT_BUTTON, self.on_upload_mhd)
        self.raw_button.Bind(wx.EVT_BUTTON, self.on_upload_raw)

        # 绑定切割和预测按钮的事件
        self.segment_button.Bind(wx.EVT_BUTTON, self.segment_and_predict)

        # 绑定可视化切片按钮的事件
        self.visualize_button.Bind(wx.EVT_BUTTON, self.visualize_slice)

        self.Centre()
        self.Show()

    def on_upload_csv(self, event):
        """上传CSV文件"""
        wildcard = "CSV files (*.csv)|*.csv"
        dialog = wx.FileDialog(self, "选择CSV文件", wildcard=wildcard, style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            csv_path = dialog.GetPath()
            self.csv_text.SetValue(csv_path)
        dialog.Destroy()

    def on_upload_mhd(self, event):
        """上传MHD文件"""
        wildcard = "MHD files (*.mhd)|*.mhd"
        dialog = wx.FileDialog(self, "选择MHD文件", wildcard=wildcard, style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            mhd_path = dialog.GetPath()
            self.mhd_text.SetValue(mhd_path)
        dialog.Destroy()

    def on_upload_raw(self, event):
        """上传RAW文件"""
        wildcard = "RAW files (*.raw)|*.raw"
        dialog = wx.FileDialog(self, "选择RAW文件", wildcard=wildcard, style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            raw_path = dialog.GetPath()
            self.raw_text.SetValue(raw_path)
        dialog.Destroy()

    def segment_and_predict(self, event):
        """进行肺结节切割并预测"""
        # 获取上传的文件路径
        csv_path = self.csv_text.GetValue()
        mhd_path = self.mhd_text.GetValue()
        raw_path = self.raw_text.GetValue()

        # 读取MHD和RAW文件
        image = self.read_mhd_raw(mhd_path, raw_path)

        # 进行肺结节切割和预测
        # 这里调用 getMat 和 readPred 函数
        fname = self.csv_text.GetValue()
        mat_path = 'data/makeMat_/'
        ct_path = os.path.dirname(self.raw_text.GetValue())
        # getMat(fname, mat_path, ct_path)

        model_path = "model/net_ALL_78.pkl"
        data_path = "data/makeMat_"
        readPred(model_path, data_path)

        # 可视化预测结果

    def visualize_slice(self, event):
        """可视化第一个切片"""
        mhd_path = self.mhd_text.GetValue()
        if os.path.exists(mhd_path):
            visualize_first_slice(mhd_path)

    def read_mhd_raw(self, mhd_path, raw_path):
        """读取MHD和RAW文件"""
        image = sitk.ReadImage(mhd_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.flipud(image_array)  # 调整方向，根据需要修改

        # 如果需要对RAW文件进行处理，可以在这里添加代码

        return image_array


if __name__ == "__main__":
    app = wx.App()
    LungNoduleDetectionApp(None, title="肺结节检测")
    app.MainLoop()
