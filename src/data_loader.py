import os
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

from . import data_augmentation
from .util import utilities as util_


def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


############# Tabletop Object Dataset #############
# Note: this is the version from "RICE: Refining Instance Masks in Cluttered
#   Environments with Graph Neural Networks", which is a slightly more cluttered
#   version that the TOD from "The Best of Both Modes: Separately Leveraging RGB
#   and Depth for Unseen Object Instance Segmentation". 


NUM_VIEWS_PER_SCENE = 5 # each directory has 5 images with objects
class Tabletop_Object_Dataset(Dataset):

    def __init__(self, base_dir, train_or_test, config):
        self.base_dir = base_dir
        self.config = config
        self.train_or_test = train_or_test

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE

        self.name = 'TableTop'

    def __len__(self):
        return self.len

    def process_rgb(self, rgb_img):
        """Process RGB image.
        """
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = data_augmentation.standardize_image(rgb_img)
        return rgb_img

    def process_depth(self, depth_img):
        """Process depth image.
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        if self.config['use_data_augmentation']:
            depth_img = data_augmentation.add_noise_to_depth(depth_img, self.config)

        # Compute xyz ordered point cloud
        xyz_img = util_.compute_xyz(depth_img, self.config)
        if self.config['use_data_augmentation']:
            xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, self.config)

        return xyz_img

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355

        # Get scene directory
        scene_idx = idx // NUM_VIEWS_PER_SCENE
        scene_dir = self.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % NUM_VIEWS_PER_SCENE + 2 # view_num=0 is always background with no table/objects

        # RGB image
        rgb_img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
        rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)

        # Depth image
        depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
        depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        xyz_img = self.process_depth(depth_img)

        # Labels
        foreground_labels_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        foreground_labels = util_.imread_indexed(foreground_labels_filename)

        label_abs_path = '/'.join(foreground_labels_filename.split('/')[-2:]) # Used for evaluation

        # predictions
        pred_filename = os.path.join(self.config['predictions_path'], label_abs_path)
        predictions = util_.imread_indexed(pred_filename)

        # Turn these all into torch tensors
        rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img) # Shape: [3 x H x W]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels) # Shape: [H x W]
        predictions = data_augmentation.array_to_tensor(predictions) # Shape: [H x W]

        return {'rgb' : rgb_img,
                'xyz' : xyz_img,
                'foreground_labels' : foreground_labels,
                'seg_masks' : predictions,
                'scene_dir' : scene_dir,
                'view_num' : view_num,
                'label_abs_path' : label_abs_path,
                }


def get_TOD_train_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = Tabletop_Object_Dataset(base_dir + 'training_set/', 'train', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

def get_TOD_test_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=False):

    config = config.copy()
    dataset = Tabletop_Object_Dataset(base_dir + 'test_set/', 'test', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

