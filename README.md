# 注视图像归一化
注视图像归一化的主要目的是消除在注视检测中，由不同头部姿态引起的大部分变异性。数据归一化首先旋转相机使眼睛图像翘曲，使相机坐标系的x轴垂直于头部坐标系的y轴。其次，对图像进行缩放，使(归一化)相机位于距离眼睛中心固定的距离。最后，对于所有不同的数据，眼睛图像在头部姿态中只有2个自由度。

本库不仅包括了注视图像归一化，还包括了一个基于XGaze baseline的检测模型。

## 相机标定
相机标定的流程如下：
1. 首先在[标定棋盘格生成网站](https://calib.io/pages/camera-calibration-pattern-generator)中生成合适的棋盘格。
2. 将棋盘格打印，或放到清晰标准的电子屏幕中，在确保棋盘格长宽与预期相符的情况下，使用摄像头以不同角度拍摄棋盘格。
3. 将棋盘格放于./test/cam，根据棋盘格修改demo_CamCal.py的各项参数，即可生成相机参数。

## gaze_normalization
为方便读取以及保存在预处理流程中使用到的各类关键变量，基于warp_norm重写了整个预处理的流程。现在，只需要创建实例对象并调用对象方法norm()即可获取归一化图片并保存相应的中间参数。此外，由于dlib人脸识别在部分条件下并不稳定，这里只使用了fa（2600张图片只有1张无法识别，包含口罩的数据集因固有缺陷，暂未得到更好的修正策略）。

## warp_norm
这个文件包括了以下在注视图像归一化中使用的函数。
### GazeNormalization
函数接受图片，相机内参矩阵(3x3)，相机畸变矩阵(1x5)，处理方式（xgaze或eve，默认为xgaze），*以下参数可不再输入*：注视坐标/注视向量，屏幕长宽，屏幕像素尺寸。返回归一化图像和注视向量（直角坐标系）。在demo_warp_norm.py中有一个简单的使用范例。
### xnorm&enorm
norm方法用于匹配人脸图像的对应点和3D人脸模型的对应点，并返回旋转向量hr和平移向量ht。xnorm是xgaze使用的方法而enorm是eve使用的方法。他们的区别在于人脸关键点的检测和选取，通用人脸模型的选择。*目前这个版本浪费时间，这里可以参考cpu branch中的做法，我随后会修改*。
### xtrans
xtrans根据hr和ht，获取对应转换后的图像，注视坐标等参数。
### draw_gaze
这个函数用于在图像上可视化注视向量，可以同时接受直角坐标系和球坐标系输入。
### pitchyaw_to_vector&inverse
这个函数用于将球面坐标和直角坐标系坐标进行转换。
### angular_error
基于函数点乘，返回两个向量的夹角大小，单位为度。
### vector_to_gc
实现向量和屏幕注视点的转换，转换的结果为相对于摄像头原点的坐标。

## 数据预处理
我的数据预处理的流程在pretreat.py中，可以根据个人路径和需要自行调整。

## XGaze baseline
接下来使用XGaze baseline进行预测。请先下载预训练模型并放入ckpt文件夹[模型下载](https://drive.google.com/file/d/1Ma6zJrECNTjo_mToZ5GKk7EF-0FS4nEC/view)
测试请参考demo_baseline_xgaze和demo_baseline_onetest。在demo_baseline_xgaze中，我将预测结果，旋转矩阵等需要使用的数据保存至gaze_pred.pkl中，result.txt只保存了预测结果。

## 数据分析（TIP）
接下来，读取数据，并使用文章中的做法进行分析，分析流程参考analyze.ipynb。目前已经完成Validity Module（*不需要使用*）和Self-Calibration Module。