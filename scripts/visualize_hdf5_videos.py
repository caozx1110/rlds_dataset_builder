import h5py
import numpy as np
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def create_videos_from_hdf5(hdf5_path, output_dir="videos", fps=10):
    """从HDF5文件创建RGB和深度图像视频"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        # 获取数据
        print("Loading data from HDF5...")
        
        # 相机名称
        cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        
        # 为每个相机创建RGB和深度视频
        for cam in cameras:
            print(f"Processing camera: {cam}")
            
            # 读取RGB图像数据
            rgb_data = f[f"observations/images/{cam}"][()]
            depth_data = f[f"observations/images_depth/{cam}"][()]
            
            num_frames = len(rgb_data)
            print(f"  Number of frames: {num_frames}")
            
            # 解码第一帧以获取尺寸
            first_rgb = Image.open(io.BytesIO(rgb_data[0]))
            first_depth = Image.open(io.BytesIO(depth_data[0]))
            
            width, height = first_rgb.size
            print(f"  Image size: {width}x{height}")
            
            # 创建RGB视频
            rgb_video_path = os.path.join(output_dir, f"{cam}_rgb.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            rgb_writer = cv2.VideoWriter(rgb_video_path, fourcc, fps, (width, height))
            
            # 创建深度视频
            depth_video_path = os.path.join(output_dir, f"{cam}_depth.mp4")
            depth_writer = cv2.VideoWriter(depth_video_path, fourcc, fps, (width, height))
            
            print(f"  Creating videos...")
            for i in range(num_frames):
                # RGB帧
                rgb_img = Image.open(io.BytesIO(rgb_data[i]))
                rgb_frame = np.array(rgb_img)
                # OpenCV使用BGR格式
                rgb_frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                rgb_writer.write(rgb_frame_bgr)
                
                # 深度帧
                depth_img = Image.open(io.BytesIO(depth_data[i]))
                depth_array = np.array(depth_img)
                
                # 将深度图归一化到0-255并转换为3通道
                depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                depth_writer.write(depth_colored)
                
                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1}/{num_frames} frames")
            
            rgb_writer.release()
            depth_writer.release()
            print(f"  Saved: {rgb_video_path}")
            print(f"  Saved: {depth_video_path}")

def create_combined_video(hdf5_path, output_dir="videos", fps=10):
    """创建包含所有视角的组合视频"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        print("Creating combined video...")
        
        cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        
        # 读取所有数据
        rgb_data = {}
        depth_data = {}
        
        for cam in cameras:
            rgb_data[cam] = f[f"observations/images/{cam}"][()]
            depth_data[cam] = f[f"observations/images_depth/{cam}"][()]
        
        num_frames = len(rgb_data[cameras[0]])
        
        # 获取单个图像尺寸
        first_img = Image.open(io.BytesIO(rgb_data[cameras[0]][0]))
        img_width, img_height = first_img.size
        
        # 组合视频尺寸：3列（3个相机），2行（RGB和深度）
        combined_width = img_width * 3
        combined_height = img_height * 2
        
        # 创建组合视频写入器
        combined_video_path = os.path.join(output_dir, "combined_all_views.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_writer = cv2.VideoWriter(combined_video_path, fourcc, fps, 
                                        (combined_width, combined_height))
        
        for i in range(num_frames):
            # 创建组合帧
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            for j, cam in enumerate(cameras):
                # RGB图像（上排）
                rgb_img = Image.open(io.BytesIO(rgb_data[cam][i]))
                rgb_array = np.array(rgb_img)
                rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                
                x_start = j * img_width
                x_end = (j + 1) * img_width
                combined_frame[0:img_height, x_start:x_end] = rgb_bgr
                
                # 深度图像（下排）
                depth_img = Image.open(io.BytesIO(depth_data[cam][i]))
                depth_array = np.array(depth_img)
                depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                
                combined_frame[img_height:2*img_height, x_start:x_end] = depth_colored
            
            # 添加文本标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)
            thickness = 2
            
            # 相机标签
            cv2.putText(combined_frame, "High", (10, 30), font, font_scale, color, thickness)
            cv2.putText(combined_frame, "Left Wrist", (img_width + 10, 30), font, font_scale, color, thickness)
            cv2.putText(combined_frame, "Right Wrist", (2 * img_width + 10, 30), font, font_scale, color, thickness)
            
            # RGB/Depth标签
            cv2.putText(combined_frame, "RGB", (10, img_height + 30), font, font_scale, color, thickness)
            cv2.putText(combined_frame, "Depth", (10, img_height + 60), font, font_scale, color, thickness)
            
            # 帧号
            cv2.putText(combined_frame, f"Frame: {i+1}/{num_frames}", 
                       (combined_width - 200, combined_height - 20), 
                       font, 0.7, color, thickness)
            
            combined_writer.write(combined_frame)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{num_frames} frames")
        
        combined_writer.release()
        print(f"Saved combined video: {combined_video_path}")

def create_matplotlib_animation(hdf5_path, output_dir="videos", fps=10, max_frames=None):
    """使用matplotlib创建动画（可选方案）"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        print("Creating matplotlib animation...")
        
        cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        
        # 读取数据
        rgb_data = {}
        depth_data = {}
        
        for cam in cameras:
            rgb_data[cam] = f[f"observations/images/{cam}"][()]
            depth_data[cam] = f[f"observations/images_depth/{cam}"][()]
        
        num_frames = len(rgb_data[cameras[0]])
        if max_frames:
            num_frames = min(num_frames, max_frames)
        
        # 解码所有帧
        rgb_frames = {}
        depth_frames = {}
        
        for cam in cameras:
            rgb_frames[cam] = []
            depth_frames[cam] = []
            
            for i in range(num_frames):
                rgb_img = Image.open(io.BytesIO(rgb_data[cam][i]))
                depth_img = Image.open(io.BytesIO(depth_data[cam][i]))
                
                rgb_frames[cam].append(np.array(rgb_img))
                depth_frames[cam].append(np.array(depth_img))
        
        # 创建matplotlib动画
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 初始化图像
        im_rgb = []
        im_depth = []
        
        for j, cam in enumerate(cameras):
            # RGB
            im_rgb.append(axes[0, j].imshow(rgb_frames[cam][0]))
            axes[0, j].set_title(f'{cam} RGB')
            axes[0, j].axis('off')
            
            # Depth
            im_depth.append(axes[1, j].imshow(depth_frames[cam][0], cmap='viridis'))
            axes[1, j].set_title(f'{cam} Depth')
            axes[1, j].axis('off')
        
        def animate(frame):
            for j, cam in enumerate(cameras):
                im_rgb[j].set_array(rgb_frames[cam][frame])
                im_depth[j].set_array(depth_frames[cam][frame])
            
            fig.suptitle(f'Frame {frame + 1}/{num_frames}')
            return im_rgb + im_depth
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000//fps, blit=True)
        
        # 保存动画
        animation_path = os.path.join(output_dir, "matplotlib_animation.mp4")
        anim.save(animation_path, writer='ffmpeg', fps=fps)
        print(f"Saved matplotlib animation: {animation_path}")
        
        plt.close()

if __name__ == "__main__":
    hdf5_path = "data/unzip_handbag/episode_8.hdf5"
    output_dir = "videos"
    
    print("Starting video creation...")
    
    # 创建单独的视频文件
    create_videos_from_hdf5(hdf5_path, output_dir, fps=30)
    
    # 创建组合视频
    create_combined_video(hdf5_path, output_dir, fps=30)

    # 可选：创建matplotlib动画（处理较少帧数）
    # create_matplotlib_animation(hdf5_path, output_dir, fps=5, max_frames=100)
    
    print("All videos created successfully!")
    print(f"Videos saved in: {output_dir}/")