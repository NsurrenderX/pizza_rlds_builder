from typing import Iterator, Tuple, Any

import os
import glob
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from scipy.spatial.transform import Rotation as R

class PizzaDatasetDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '2.0.0': 'Task 2.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
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
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/datahdd_8T/pizza_dataset/', id='1'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path, id= '1') -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        
            
        def _parse_example(episode_path):
            def resize_image(image):
                #crop the image into square
                w, h ,_ = image.shape
                if w > h:
                    image = image[int((w-h)/2):int((w-h)/2)+h, :]
                else:
                    # -75 is to make sure the platform in the middle
                    image = image[:, int((h-w)/2)-75:int((h-w)/2)+w-75, :]
                image = cv2.resize(image, (224, 224))
                return image
            
            def action_to_euler(action):
                # convert npy robot state to 7dof action vector
                action_vector = np.zeros(7).astype(np.float32)
                # action measured in meters
                action_vector[0:3] = action['eef_position']
                # convert action quaternion to euler angle
                action_vector[3:6] = R.from_quat(action['eef_quaternion']).as_euler('xyz', degrees=False)
                action_vector[6] = action['gripper_width']
                
                # flip the action vector in (y, z) axis to match the standard
                action_vector[1] = -action_vector[1]
                action_vector[2] = -action_vector[2]
                return action_vector
        
            def get_language_instruction(path):
                # get language instruction from the prompt file
                task_id = int(path.split('/')[-2])
                prompt_path = os.path.join(path, '..', '..',  'prompts.txt')
                with open(prompt_path, 'r') as f:
                    tasks = []
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('Task'):
                            tasks.append(line.split(': ')[-1].strip())
                print(tasks)
                return tasks[task_id-1]
            
            def get_diff_data(raw_data):
                # get the action matrix from collected robot states
                total_diff = len(raw_data) - 1
                data_mat = []
                for i in range(len(raw_data)):
                    data_mat.append(action_to_euler(raw_data[i]))
                diff_mat = []
                for i in range(total_diff):
                    raw_diff_vec = data_mat[i + 1] - data_mat[i]
                    for i in range(3):
                        if raw_diff_vec[i+3] > np.pi:
                            raw_diff_vec[i+3] = raw_diff_vec[i+3] - 2*np.pi
                        elif raw_diff_vec[i+3] < -np.pi:
                            raw_diff_vec[i+3] = raw_diff_vec[i+3] + 2*np.pi
                    diff_mat.append(raw_diff_vec)
                return diff_mat
            
            language_instruction = get_language_instruction(episode_path)
            npy_path = os.path.join(episode_path, 'franka_data.npy')
            raw_data = np.load(npy_path, allow_pickle=True)     # this is a list of dicts in our case
            images_path = os.path.join(episode_path, 'images', 'right_rgb')
            images = os.listdir(images_path)
            img_ids = []
            for img_id in images:
                img_ids.append(int(img_id[:-4]))
            img_ids.sort()
            images = [f'{i:03d}.jpg' for i in img_ids]
            
            range_action = len(raw_data)
            
            if range_action <= int(images[-1][:-4]):
                if os.path.exists(os.path.join(images_path, f'{(range_action - 1):03d}.jpg')):
                    range_index = images.index(f'{(range_action - 1):03d}.jpg')
                    images = images[1:range_index]
                else:
                    for i in range(range_action, 0, -1):
                        if os.path.exists(os.path.join(images_path, f'{i:03d}.jpg')):
                            range_action = i
                            range_index = images.index(f'{i:03d}.jpg')
                            break
                    images = images[1:range_index]
            else:
                images = images[1:]
            data = get_diff_data(raw_data)

            print(data[1]) #check data format
           
            episode = []
            prev = 0
            
            for i, image in enumerate(images):
                
                # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()
                step_len = int(image[:-4])
                
                action = data[0]
                
                for k in range(prev, step_len):
                    print(f"k: {k} raw_data: {len(raw_data)} data: {len(data)} images: {len(images)} episode_path: {episode_path} final_image_id : {images[-1]}")
                    action = action + data[k]
                action = action - data[0]
                action[1] = -action[1]
                action[2] = -action[2]
                
                image_array = cv2.imread(os.path.join(images_path, f'{prev:03d}.jpg'))
                image_array = resize_image(image_array)
                print(image_array.shape, action, language_instruction)
                episode.append({
                    'observation': {
                        'image': image_array,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

                # create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': episode_path
                    }
                }
                prev = int(image[:-4])

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample
        # print("exec here")
        # create list of all examples
        sample_paths = sorted(os.listdir(path))
        episode_paths = []
        for task in sample_paths:
            if task == '_init_.py' or task == '__pycache__' or task == 'pizza_dataset_dataset_builder.py' or task == 'prompts.txt':
                continue
            if task == id:
                path_sins = sorted(os.listdir(os.path.join(path, task)))
            else:
                path_sins = []
            for path_sin in path_sins:
            
                npy_path = os.path.join(path, task, path_sin, 'franka_data.npy')
                image_path = os.path.join(path, task, path_sin,'images', 'right_rgb')
                if os.path.exists(npy_path) and os.path.exists(image_path):
                    episode_paths.append(os.path.join(path, task, path_sin))
            # for smallish datasets, use single-thread parsing
        print("episode_paths:", episode_paths)
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

