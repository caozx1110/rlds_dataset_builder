import h5py
from PIL import Image
import io

def print_hdf5_structure(file_path):
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_attrs)

def inspect_image_shape(hdf5_path, dataset_path):
    with h5py.File(hdf5_path, 'r') as f:
        img_bytes = f[dataset_path][0]  # 取第一帧
        img = Image.open(io.BytesIO(img_bytes))
        print(f"{dataset_path} 第一帧 shape: {img.size}, mode: {img.mode}")

if __name__ == "__main__":
    hdf5_path = "data/episode_72.hdf5"
    print_hdf5_structure(hdf5_path)
    for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        inspect_image_shape(hdf5_path, f"observations/images/{cam}")
        inspect_image_shape(hdf5_path, f"observations/images_depth/{cam}")