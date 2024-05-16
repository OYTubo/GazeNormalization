# 输入要求：

1. 相机矩阵：.yaml格式    默认是./testpart/kinect-intrinsics.yaml
2. 检测视频：.mp4视频，默认是./testpart/3-1.mp4

# 输出格式：

1. 检测完成的视频：.mp4视频，保存在./test/output_3-2.mp4
（1) 视频左上角3D箭头代表屏幕坐标系的注视向量
（2）alert：疲劳警告，条件是a)眨眼时长大于等于5帧的占一分钟窗口内的25%以上;b)出现眨眼时长大于10帧的，可在returnAlert函数中修改条件
（3）blinkLonggest：最长眨眼次数
（4）frameCount：总眨眼帧数
（5）blinkCount：总眨眼次数
（6）EAR：眼睛纵横比
2.Ear：眼睛纵横比  每帧一个  大小[0,1]  print输出
3.Elapsed time： 运行时间  单位秒  print输出
4.Predict normalization gaze vector(pitch yaw)：预测的规范化注视向量(俯仰角和偏航角)  每帧一列(二维向量)  大小([0,pi],[0,2pi])
5.pred vector：预测的注视向量  每帧一列(三维单位向量)  （e1,e2,e3)  ||e|| = 1
6.Normalization pred gaze point：预测的规范化注视点  每帧一列  随pixel_scale变化