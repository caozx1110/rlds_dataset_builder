
# RLDS Dataset Conversion

## RHOS ALOHA数据集转换指南

### 概述

本节介绍如何将RHOS ALOHA数据集转换为RLDS格式。**重要说明**：一次只能生成一个instruction对应的数据文件夹，不能同时转换多个子数据集。

### 数据集存放结构

原始数据集应按以下结构存放：

```
data/
├── unzip_handbag/           # 单个任务文件夹（一次只处理一个）
│   ├── episode_8.hdf5      # HDF5数据文件
│   ├── episode_72.hdf5     # 更多episode文件
│   ├── ...
│   └── metadata.json       # 包含task_description_english的元数据
├── put_marker_in_bag/       # 其他任务文件夹（需要单独处理）
│   ├── episode_1.hdf5
│   ├── episode_2.hdf5
│   └── metadata.json
└── ...
```

### metadata.json格式

每个任务文件夹必须包含metadata.json文件：

```json
{
    "task_description_english": "Unzip the bag, put the marker in the bag, and zip it up.",
    "nodes": [...]
}
```

### 转换步骤

#### 1. 环境准备

```bash
conda env create -f environment_ubuntu.yml
conda activate rlds_env
```

#### 2. 修改数据集路径

编辑 `rhos_aloha/rhos_aloha_dataset_builder.py` 中的 `_split_paths()` 方法：

**单个任务转换**：

```python
def _split_paths(self):
    """Define filepaths for data splits."""
    # 一次只处理一个任务文件夹
    return {
        "train": glob.glob("/path/to/data/unzip_handbag/*.hdf5")
    }
```

**切换到其他任务**时，需要修改路径：

```python
def _split_paths(self):
    """Define filepaths for data splits."""
    # 切换到不同任务需要修改这里
    return {
        "train": glob.glob("/path/to/data/put_marker_in_bag/*.hdf5")
    }
```

#### 3. 执行转换

```bash
cd rhos_aloha
tfds build --overwrite
```

#### 4. 验证转换结果

```bash
# 从项目根目录运行
python3 scripts/verify_rlds_conversion.py
```

#### 5. 可视化数据

```bash
# 生成HDF5原始数据的视频
python3 scripts/visualize_hdf5_videos.py

# 可视化转换后的RLDS数据集
python3 visualize_dataset.py rhos_aloha
```

### 重要注意事项

1. **单任务处理**：每次转换只能处理一个instruction对应的数据文件夹，如果要转换多个任务，需要：

   - 修改 `_split_paths()` 中的路径
   - 重新运行 `tfds build --overwrite`
   - 每个任务会生成独立的数据集
2. **数据格式要求**：

   - HDF5文件必须包含：action, observations/qpos, observations/qvel, observations/effort
   - 图像数据：observations/images/{cam_high, cam_left_wrist, cam_right_wrist}
   - 深度数据：observations/images_depth/{cam_high, cam_left_wrist, cam_right_wrist}
   - 图像格式：RGB(480,640,3), Depth(480,640,1)
3. **任务描述**：每个episode会从同目录的metadata.json中读取task_description_english作为language_instruction
4. **输出位置**：转换后的数据集保存在 `~/tensorflow_datasets/rhos_aloha/`

### 数据集特征

- **观测空间**：3个RGB相机 + 3个深度相机 + 机器人状态(qpos, qvel, effort)
- **动作空间**：14维机器人关节动作
- **语言指令**：从metadata.json读取的任务描述
- **Episode长度**：通常400+步

---

# RLDS Dataset Conversion

This repo demonstrates how to convert an existing dataset into RLDS format for X-embodiment experiment integration.
It provides an example for converting a dummy dataset to RLDS. To convert your own dataset, **fork** this repo and
modify the example code for your dataset following the steps below.

## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):

```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:

```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`,
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly` and `wandb`.

## Run Example RLDS Dataset Creation

Before modifying the code to convert your own dataset, run the provided example dataset creation script to ensure
everything is installed correctly. Run the following lines to create some dummy data and convert it to RLDS.

```
cd example_dataset
python3 create_example_data.py
tfds build
```

This should create a new dataset in `~/tensorflow_datasets/example_dataset`. Please verify that the example
conversion worked before moving on.

## Converting your Own Dataset to RLDS

Now we can modify the provided example to convert your own data. Follow the steps below:

1. **Rename Dataset**: Change the name of the dataset folder from `example_dataset` to the name of your dataset (e.g. robo_net_v2),
   also change the name of `example_dataset_dataset_builder.py` by replacing `example_dataset` with your dataset's name (e.g. robo_net_v2_dataset_builder.py)
   and change the class name `ExampleDataset` in the same file to match your dataset's name, using camel case instead of underlines (e.g. RoboNetV2).
2. **Modify Features**: Modify the data fields you plan to store in the dataset. You can find them in the `_info()` method
   of the `ExampleDataset` class. Please add **all** data fields your raw data contains, i.e. please add additional features for
   additional cameras, audio, tactile features etc. If your type of feature is not demonstrated in the example (e.g. audio),
   you can find a list of all supported feature types [here](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?hl=en#classes).
   You can store step-wise info like camera images, actions etc in `'steps'` and episode-wise info like `collector_id` in `episode_metadata`.
   Please don't remove any of the existing features in the example (except for `wrist_image` and `state`), since they are required for RLDS compliance.
   Please add detailed documentation what each feature consists of (e.g. what are the dimensions of the action space etc.).
   Note that we store `language_instruction` in every step even though it is episode-wide information for easier downstream usage (if your dataset
   does not define language instructions, you can fill in a dummy string like `pick up something`).
3. **Modify Dataset Splits**: The function `_split_generator()` determines the splits of the generated dataset (e.g. training, validation etc.).
   If your dataset defines a train vs validation split, please provide the corresponding information to `_generate_examples()`, e.g.
   by pointing to the corresponding folders (like in the example) or file IDs etc. If your dataset does not define splits,
   remove the `val` split and only include the `train` split. You can then remove all arguments to `_generate_examples()`.
4. **Modify Dataset Conversion Code**: Next, modify the function `_generate_examples()`. Here, your own raw data should be
   loaded, filled into the episode steps and then yielded as a packaged example. Note that the value of the first return argument,
   `episode_path` in the example, is only used as a sample ID in the dataset and can be set to any value that is connected to the
   particular stored episode, or any other random value. Just ensure to avoid using the same ID twice.
5. **Provide Dataset Description**: Next, add a bibtex citation for your dataset in `CITATIONS.bib` and add a short description
   of your dataset in `README.md` inside the dataset folder. You can also provide a link to the dataset website and please add a
   few example trajectory images from the dataset for visualization.
6. **Add Appropriate License**: Please add an appropriate license to the repository.
   Most common is the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license --
   you can copy it from [here](https://github.com/teamdigitale/licenses/blob/master/CC-BY-4.0).

That's it! You're all set to run dataset conversion. Inside the dataset directory, run:

```
tfds build --overwrite
```

The command line output should finish with a summary of the generated dataset (including size and number of samples).
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/<name_of_your_dataset>`.

### Parallelizing Data Processing

By default, dataset conversion is single-threaded. If you are parsing a large dataset, you can use parallel processing.
For this, replace the last two lines of `_generate_examples()` with the commented-out `beam` commands. This will use
Apache Beam to parallelize data processing. Before starting the processing, you need to install your dataset package
by filling in the name of your dataset into `setup.py` and running `pip install -e .`

Then, make sure that no GPUs are used during data processing (`export CUDA_VISIBLE_DEVICES=`) and run:

```
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```

You can specify the desired number of workers with the `direct_num_workers` argument.

## Visualize Converted Dataset

To verify that the data is converted correctly, please run the data visualization script from the base directory:

```
python3 visualize_dataset.py <name_of_your_dataset>
```

This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and
add your own WandB entity -- then the script will log all visualizations to WandB.

## Add Transform for Target Spec

For X-embodiment training we are using specific inputs / outputs for the model: input is a single RGB camera, output
is an 8-dimensional action, consisting of end-effector position and orientation, gripper open/close and a episode termination
action.

The final step in adding your dataset to the training mix is to provide a transform function, that transforms a step
from your original dataset above to the required training spec. Please follow the two simple steps below:

1. **Modify Step Transform**: Modify the function `transform_step()` in `example_transform/transform.py`. The function
   takes in a step from your dataset above and is supposed to map it to the desired output spec. The file contains a detailed
   description of the desired output spec.
2. **Test Transform**: We provide a script to verify that the resulting __transformed__ dataset outputs match the desired
   output spec. Please run the following command: `python3 test_dataset_transform.py <name_of_your_dataset>`

If the test passes successfully, you are ready to upload your dataset!

## Upload Your Data

We provide a Google Cloud bucket that you can upload your data to. First, install `gsutil`, the Google cloud command
line tool. You can follow the installation instructions [here](https://cloud.google.com/storage/docs/gsutil_install).

Next, authenticate your Google account with:

```
gcloud auth login
```

This will open a browser window that allows you to log into your Google account (if you're on a headless server,
you can add the `--no-launch-browser` flag). Ideally, use the email address that
you used to communicate with Karl, since he will automatically grant permission to the bucket for this email address.
If you want to upload data with a different email address / google account, please shoot Karl a quick email to ask
to grant permissions to that Google account!

After logging in with a Google account that has access permissions, you can upload your data with the following
command:

```
gsutil -m cp -r ~/tensorflow_datasets/<name_of_your_dataset> gs://xembodiment_data
```

This will upload all data using multiple threads. If your internet connection gets interrupted anytime during the upload
you can just rerun the command and it will resume the upload where it was interrupted. You can verify that the upload
was successful by inspecting the bucket [here](https://console.cloud.google.com/storage/browser/xembodiment_data).

The last step is to commit all changes to this repo and send Karl the link to the repo.

**Thanks a lot for contributing your data!
