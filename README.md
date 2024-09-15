# RLDS Databuilder for Pizza Dataset 

## Data Statement
This pizza dataset features 22 tasks about cooking pizza from scratch. The dataset is collected using a franka robot arm and four cameras. Most of the data was recorded at a rate of 5Hz, except for tasks 3, 19, and 20, which were sampled at 6Hz.

In the root folder, the "prompts.txt" file provides the prompts for all tasks in both Chinese and English. Please note that only the first line of each task is used as the task description within the dataset. The remaining lines serve as subtasks for the specified task.

Our raw data is organized by task. For instance, folder "4" contains all data collected during the execution of Task 4. Within each task folder, the data for each episode is stored in a separate folder, and the robot state data for each episode is saved in a file named "franka_data.npy." 
Below is the structure of the dataset:

```
pizza_dataset
├── prompts.txt
├── task_id
│   ├── episode_id
│   │   ├── franka_data.npy
│   │   ├── images
│   │   │   ├── inhand_rgb
│   │   │   │   ├──{id:03d}.jpg //id represents the frame id, the first frame is "000.jpg"
│   │   │   │   ├── ...
│   │   │   ├── top_rgb
│   │   │   │   ├── ...
│   │   │   ├── right_rgb
│   │   │   │   ├── ...
│   │   │   ├── left_rgb
│   │   │   │   ├── ...
├── ...
```

#### The npy file contains the following items:

| Item Name | Dimension | Description |
| --- | --- | --- |
| timestamp | 1 | The timestamp of the data. |
| eef_position | 3 | The positions of the robot arm's end effector, displayed as (x,y,z) vectors, measured in meters. |
| eef_quaternion | 4 | The quaternions of the robot arm's end effector, presented as 4-dimensional vectors, can be converted to 3-dimensional angle data. |
| joint_torque | 7 | The joint torques of the robot arm. |
| joint_velocity | 7 | The joint velocities of the robot arm. |
| gripper_width | 1 | The width of the robot arm's gripper, measured in meters |

## Usage
This repository contains two files: pizza_dataset_dataset_builder.py and rlds_builder.yaml. The rlds_builder.yaml file is a conda environment configuration file, which can be used to set up the corresponding conda environment. The pizza_dataset_dataset_builder.py is a Python script that converts our pizza dataset into the RLDS format. For specific usage, you can refer to [this repository](https://github.com/kpertsch/rlds_dataset_builder). The Python file corresponds to the example_dataset_dataset_builder.py found in the example_dataset folder of the rlds_dataset_builder repository.

The default usage is as follows:
```
1) Copy pizza_dataset_dataset_builder.py to the current directory in your bash terminal
2) Activate the conda environment created by rlds_builder.yaml
3) Modify the 'path' parameter at line 84 of pizza_dataset_dataset_builder.py to point to the folder containing our pizza data
4) Modify the 'id' parameter at line 84 of pizza_dataset_dataset_builder.py to point to the task ID you want to convert
5) tfds build --data_dir "path/to/save/dir"
```
After the conversion process, you will obtain the dataset in RLDS format corresponding to the task specified by the 'id'.


## Caution
#### The data in this dataset is raw and requires alignment and preprocessing before further usage. Usually the first state is considered aligned with the first frame of each episode.
