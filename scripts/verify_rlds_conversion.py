import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def load_and_verify_rlds_dataset(dataset_name="rhos_aloha"):
    """加载并验证RLDS数据集转换的正确性"""
    
    # 加载数据集
    print(f"Loading dataset: {dataset_name}")
    ds = tfds.load(dataset_name, split='train')
    
    # 获取第一个episode
    episode = next(iter(ds))
    
    print("="*50)
    print("EPISODE METADATA:")
    print("="*50)
    for key, value in episode['episode_metadata'].items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50)
    print("EPISODE STRUCTURE:")
    print("="*50)
    
    steps = episode['steps']
    num_steps = len(steps)
    print(f"Number of steps in episode: {num_steps}")
    
    # 检查第一个step的结构
    first_step = next(iter(steps))
    
    print("\nFirst step structure:")
    for key, value in first_step.items():
        if key == 'observation':
            print(f"  {key}:")
            for obs_key, obs_value in value.items():
                print(f"    {obs_key}: shape={obs_value.shape}, dtype={obs_value.dtype}")
        else:
            print(f"  {key}: shape={value.shape if hasattr(value, 'shape') else 'scalar'}, dtype={value.dtype}, value={value}")

def visualize_episode_data(dataset_name="rhos_aloha", step_indices=[0, 50, 100, 200, 400]):
    """可视化episode中的图像和数据"""
    
    ds = tfds.load(dataset_name, split='train')
    episode = next(iter(ds))
    steps = list(episode['steps'])
    
    print(f"\nVisualizing steps: {step_indices}")
    
    # 创建图像可视化
    fig, axes = plt.subplots(len(step_indices), 6, figsize=(20, 4*len(step_indices)))
    if len(step_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, step_idx in enumerate(step_indices):
        if step_idx >= len(steps):
            print(f"Warning: Step {step_idx} exceeds episode length {len(steps)}")
            continue
            
        step = steps[step_idx]
        obs = step['observation']
        
        # RGB images
        axes[i, 0].imshow(obs['image'].numpy())
        axes[i, 0].set_title(f'Step {step_idx}: Main Camera')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(obs['left_wrist_image'].numpy())
        axes[i, 1].set_title(f'Step {step_idx}: Left Wrist')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(obs['right_wrist_image'].numpy())
        axes[i, 2].set_title(f'Step {step_idx}: Right Wrist')
        axes[i, 2].axis('off')
        
        # Depth images
        depth_main = obs['depth_image'].numpy().squeeze()
        axes[i, 3].imshow(depth_main, cmap='viridis')
        axes[i, 3].set_title(f'Step {step_idx}: Main Depth')
        axes[i, 3].axis('off')
        
        depth_left = obs['left_wrist_depth_image'].numpy().squeeze()
        axes[i, 4].imshow(depth_left, cmap='viridis')
        axes[i, 4].set_title(f'Step {step_idx}: Left Depth')
        axes[i, 4].axis('off')
        
        depth_right = obs['right_wrist_depth_image'].numpy().squeeze()
        axes[i, 5].imshow(depth_right, cmap='viridis')
        axes[i, 5].set_title(f'Step {step_idx}: Right Depth')
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig('images/rlds_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_action_trajectories(dataset_name="rhos_aloha"):
    """绘制动作轨迹"""
    
    ds = tfds.load(dataset_name, split='train')
    episode = next(iter(ds))
    steps = list(episode['steps'])
    
    # 提取所有动作
    actions = np.array([step['action'].numpy() for step in steps])
    qpos = np.array([step['observation']['qpos'].numpy() for step in steps])
    qvel = np.array([step['observation']['qvel'].numpy() for step in steps])
    
    print(f"\nAction trajectory shape: {actions.shape}")
    print(f"Qpos trajectory shape: {qpos.shape}")
    print(f"Qvel trajectory shape: {qvel.shape}")
    
    # 绘制动作轨迹
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Actions
    for i in range(min(14, actions.shape[1])):
        axes[0].plot(actions[:, i], label=f'action_{i}', alpha=0.7)
    axes[0].set_title('Action Trajectories')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Action Value')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True)
    
    # Joint positions
    for i in range(min(14, qpos.shape[1])):
        axes[1].plot(qpos[:, i], label=f'qpos_{i}', alpha=0.7)
    axes[1].set_title('Joint Position Trajectories')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Position')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True)
    
    # Joint velocities
    for i in range(min(14, qvel.shape[1])):
        axes[2].plot(qvel[:, i], label=f'qvel_{i}', alpha=0.7)
    axes[2].set_title('Joint Velocity Trajectories')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Velocity')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('images/action_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_original_vs_rlds(original_hdf5_path="data/episode_72.hdf5", dataset_name="rhos_aloha"):
    """比较原始HDF5数据和转换后的RLDS数据"""
    import h5py
    
    print("="*50)
    print("COMPARING ORIGINAL vs RLDS DATA:")
    print("="*50)
    
    # 读取原始数据
    with h5py.File(original_hdf5_path, 'r') as f:
        orig_actions = f['/action'][()]
        orig_qpos = f['/observations/qpos'][()]
        orig_qvel = f['/observations/qvel'][()]
        orig_effort = f['/observations/effort'][()]
    
    # 读取RLDS数据
    ds = tfds.load(dataset_name, split='train')
    episode = next(iter(ds))
    steps = list(episode['steps'])
    
    rlds_actions = np.array([step['action'].numpy() for step in steps])
    rlds_qpos = np.array([step['observation']['qpos'].numpy() for step in steps])
    rlds_qvel = np.array([step['observation']['qvel'].numpy() for step in steps])
    rlds_effort = np.array([step['observation']['effort'].numpy() for step in steps])
    
    # 比较数据
    print(f"Original actions shape: {orig_actions.shape}")
    print(f"RLDS actions shape: {rlds_actions.shape}")
    print(f"Actions match: {np.allclose(orig_actions, rlds_actions)}")
    
    print(f"\nOriginal qpos shape: {orig_qpos.shape}")
    print(f"RLDS qpos shape: {rlds_qpos.shape}")
    print(f"Qpos match: {np.allclose(orig_qpos, rlds_qpos)}")
    
    print(f"\nOriginal qvel shape: {orig_qvel.shape}")
    print(f"RLDS qvel shape: {rlds_qvel.shape}")
    print(f"Qvel match: {np.allclose(orig_qvel, rlds_qvel)}")
    
    print(f"\nOriginal effort shape: {orig_effort.shape}")
    print(f"RLDS effort shape: {rlds_effort.shape}")
    print(f"Effort match: {np.allclose(orig_effort, rlds_effort)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify RLDS dataset conversion and visualize data.")
    parser.add_argument('--original_hdf5_path', type=str, default="/home/ubuntu/ws/rlds_dataset_builder/data/episode_72.hdf5",
                        help="Path to the original HDF5 file.")
    parser.add_argument('--dataset_name', type=str, default="rhos_aloha",
                        help="Name of the converted RLDS dataset (TFDS registered name).")
    args = parser.parse_args()

    try:
        # 验证数据集结构
        load_and_verify_rlds_dataset(dataset_name=args.dataset_name)
        
        # 可视化图像数据
        visualize_episode_data(dataset_name=args.dataset_name, step_indices=[0, 100, 200, 300, 400])
        
        # 绘制动作轨迹
        plot_action_trajectories(dataset_name=args.dataset_name)
        
        # 比较原始数据和转换后数据
        compare_original_vs_rlds(original_hdf5_path=args.original_hdf5_path, dataset_name=args.dataset_name)
        
        print("\n" + "="*50)
        print("VERIFICATION COMPLETE!")
        print("="*50)
        
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()