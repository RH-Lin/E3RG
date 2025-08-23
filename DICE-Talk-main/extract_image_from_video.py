# extract_middle_frame.py
# 提取指定文件夹下每个视频的中间帧，并保存为图片

import os
import glob
import cv2

# 输入视频路径和输出图片路径
video_dir = "datasets/AvaMERG/test/test_video"
image_dir = "infer_outputs_avamerg/avamerg_test_images"

# 支持的视频文件扩展名
video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')

# 创建输出目录（如果不存在）
os.makedirs(image_dir, exist_ok=True)

# 遍历所有视频文件
for ext in video_extensions:
    pattern = os.path.join(video_dir, ext)
    for video_path in glob.glob(pattern):
        # 获取视频文件名，无扩展名
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue

        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"无法获取帧数: {video_path}")
            cap.release()
            continue

        # 计算中间帧索引
        mid_frame_idx = 1 #total_frames // 2

        # 设置视频帧位置到中间帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)

        # 读取这一帧
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取中间帧: {video_path}")
            cap.release()
            continue

        # 保存图像，使用jpg格式，可根据需要改成png
        output_path = os.path.join(image_dir, f"{video_name}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"已保存: {output_path}")

        # 释放资源
        cap.release()
