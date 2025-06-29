import h5py
from PIL import Image
import io
import numpy as np

def check_all_image_dimensions(hdf5_path):
    """检查所有图像的实际维度"""
    with h5py.File(hdf5_path, 'r') as f:
        print("检查所有图像的维度:")
        
        # RGB图像
        for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            rgb_data = f[f"observations/images/{cam}"]
            img_bytes = rgb_data[0]
            img = Image.open(io.BytesIO(img_bytes))
            print(f"RGB {cam}: PIL size={img.size}, mode={img.mode}")
            
            # 转换为numpy检查shape
            img_array = np.array(img)
            print(f"RGB {cam}: numpy shape={img_array.shape}")
        
        print()
        
        # 深度图像
        for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            depth_data = f[f"observations/images_depth/{cam}"]
            img_bytes = depth_data[0]
            img = Image.open(io.BytesIO(img_bytes))
            print(f"Depth {cam}: PIL size={img.size}, mode={img.mode}")
            
            # 转换为numpy检查shape
            img_array = np.array(img)
            print(f"Depth {cam}: numpy shape={img_array.shape}")

if __name__ == "__main__":
    check_all_image_dimensions("data/episode_72.hdf5")