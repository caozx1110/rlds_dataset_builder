from typing import Iterator, Tuple, Any
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import os
import json
from PIL import Image
import io

sys.path.append('.')
from rhos_aloha.conversion_utils import MultiThreadedDatasetBuilder
import glob

def _decode_image(img_bytes, is_depth=False):
    """解码图像字节为numpy数组"""
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img)
    
    if is_depth:
        # 深度图像：确保是(H, W, 1)格式
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
    else:
        # RGB图像：确保是(H, W, 3)格式
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
    
    return img_array

def _load_task_description(episode_path):
    """从metadata.json文件中加载任务描述"""
    # 获取episode文件所在目录
    episode_dir = os.path.dirname(episode_path)
    metadata_path = os.path.join(episode_dir, "metadata.json")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get("task_description_english", "Unknown task")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        print(f"Warning: Could not load task description from {metadata_path}")
        return "Unknown task"

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    def _parse_example(episode_path):
        # 加载任务描述
        task_description = _load_task_description(episode_path)
        
        with h5py.File(episode_path, "r") as F:
            actions = F["/action"][()]  # (T, 14)
            qpos = F["/observations/qpos"][()]  # (T, 14)
            qvel = F["/observations/qvel"][()]  # (T, 14)
            effort = F["/observations/effort"][()]  # (T, 14)
            # 图像（原始字节）
            cam_high = F["/observations/images/cam_high"][()]  # (T,)
            cam_left_wrist = F["/observations/images/cam_left_wrist"][()]  # (T,)
            cam_right_wrist = F["/observations/images/cam_right_wrist"][()]  # (T,)
            # 深度图（原始字节）
            depth_high = F["/observations/images_depth/cam_high"][()]
            depth_left_wrist = F["/observations/images_depth/cam_left_wrist"][()]
            depth_right_wrist = F["/observations/images_depth/cam_right_wrist"][()]

        episode = []
        for i in range(actions.shape[0]):
            # 解码图像
            rgb_high = _decode_image(cam_high[i], is_depth=False)
            rgb_left = _decode_image(cam_left_wrist[i], is_depth=False) 
            rgb_right = _decode_image(cam_right_wrist[i], is_depth=False)
            
            depth_high_img = _decode_image(depth_high[i], is_depth=True)
            depth_left_img = _decode_image(depth_left_wrist[i], is_depth=True)
            depth_right_img = _decode_image(depth_right_wrist[i], is_depth=True)
            
            episode.append({
                'observation': {
                    'image': rgb_high.astype(np.uint8),
                    'left_wrist_image': rgb_left.astype(np.uint8),
                    'right_wrist_image': rgb_right.astype(np.uint8),
                    'depth_image': depth_high_img.astype(np.uint16),
                    'left_wrist_depth_image': depth_left_img.astype(np.uint16),
                    'right_wrist_depth_image': depth_right_img.astype(np.uint16),
                    'qpos': np.asarray(qpos[i], np.float32),
                    'qvel': np.asarray(qvel[i], np.float32),
                    'effort': np.asarray(effort[i], np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': task_description,  # 添加语言指令
            })

        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path,
                'task_description': task_description,  # 添加到元数据中
            }
        }
        return episode_path, sample

    for sample in paths:
        ret = _parse_example(sample)
        yield ret

class rhos_aloha(MultiThreadedDatasetBuilder):
    """DatasetBuilder for rhos_aloha dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 4
    MAX_PATHS_IN_MEMORY = 10
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),  # 修正为实际的高x宽
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'left_wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Left wrist camera RGB observation.',
                        ),
                        'right_wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right wrist camera RGB observation.',
                        ),
                        'depth_image': tfds.features.Image(
                            shape=(480, 640, 1),  # 修正为实际的高x宽
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='Main camera depth observation.',
                        ),
                        'left_wrist_depth_image': tfds.features.Image(
                            shape=(480, 640, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='Left wrist camera depth observation.',
                        ),
                        'right_wrist_depth_image': tfds.features.Image(
                            shape=(480, 640, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='Right wrist camera depth observation.',
                        ),
                        'qpos': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot joint positions.',
                        ),
                        'qvel': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot joint velocities.',
                        ),
                        'effort': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot joint efforts.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot arm action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount, default 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction for the task.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'task_description': tfds.features.Text(
                        doc='Task description from metadata.json.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("/inspire/hdd/project/robot-hardware/public/rhos_data/unzip_handbag/task_00017_user_00004_scene_00001/*.hdf5"),
            "val": glob.glob("/inspire/hdd/project/robot-hardware/public/rhos_data/unzip_handbag/task_00017_user_00004_scene_00001/*.hdf5")
        }
        # return {
        #     "train": ["/home/ubuntu/ws/rlds_dataset_builder/data/unzip_handbag/episode_8.hdf5"]
        # }
