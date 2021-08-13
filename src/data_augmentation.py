import torch
import random
import numpy as np
import cv2

from .util import utilities as util_


##### Useful Utilities #####

def array_to_tensor(array):
    """Convert a numpy.ndarray to torch.FloatTensor.

    numpy.ndarray [N, H, W, C] -> torch.FloatTensor [N, C, H, W]
        OR
    numpy.ndarray [H, W, C] -> torch.FloatTensor [C, H, W]
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor


##### Depth augmentations #####

def add_noise_to_depth(depth_img, noise_params):
    """Add noise to depth image. 
    
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

    Args:
        depth_img: a [H, W] numpy.ndarray set of depth z values in meters.

    Returns:
        a [H, W] numpy.ndarray.
    """
    depth_img = depth_img.copy()

    # Multiplicative noise: Gamma random variable
    multiplicative_noise = np.random.gamma(noise_params['gamma_shape'], noise_params['gamma_scale'])
    depth_img = multiplicative_noise * depth_img

    return depth_img

def add_noise_to_xyz(xyz_img, depth_img, noise_params):
    """Add (approximate) Gaussian Process noise to ordered point cloud.

    Args:
        xyz_img: a [H, W, 3] ordered point cloud.

    Returns:
        a [H, W, 3] ordered point cloud.
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    gp_rescale_factor = np.random.randint(noise_params['gp_rescale_factor_range'][0],
                                          noise_params['gp_rescale_factor_range'][1])
    gp_scale = np.random.uniform(noise_params['gaussian_scale_range'][0],
                                 noise_params['gaussian_scale_range'][1])

    small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
    additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

    return xyz_img


##### RGB Augmentations #####

def standardize_image(image):
    """Standardize RGB image.

    Subtract ImageNet mean and divide by ImageNet stddev.
    Convert a numpy.ndarray [H, W, 3] RGB image to [0,1] range, and then standardizes.

    Args:
        image: a [H, W, 3] np.ndarray RGB image.

    Returns:
        a [H, W, 3] numpy array of np.float32.
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized

def unstandardize_image(image):
    """Convert standardized image back to RGB.

    Inverse of standardize_image()

    Args:
        image: a [H, W, 3] np.ndarray RGB image.

    Returns:
        a [H, W, 3] numpy array of type np.uint8.
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_img = (image * std[None,None,:] + mean[None,None,:]) * 255.
    return orig_img.round().astype(np.uint8)


##### Label transformations #####

def random_rotation(label, noise_params):
    """ Randomly rotate mask

        @param label: a [H, W] numpy array of {0, 1}
    """
    H, W = label.shape

    num_tries = 0
    valid_transform = False
    while not valid_transform:

        if num_tries >= noise_params['max_augmentation_tries']:
            print('Rotate: Exhausted number of augmentation tries...')
            return label

        # Rotate about center of box
        pixel_indices = util_.build_matrix_of_indices(H, W)
        h_idx, w_idx = np.where(label)
        mean = np.mean(pixel_indices[h_idx, w_idx, :], axis=0) # Shape: [2]. y_center, x_center

        # Sample an angle
        applied_angle = np.random.uniform(-noise_params['rotation_angle_max'], 
                                           noise_params['rotation_angle_max'])

        rotated_label = rotate(label, applied_angle, center=tuple(mean[::-1]), interpolation=cv2.INTER_NEAREST)

        # Make sure the mass is reasonable
        if (np.count_nonzero(rotated_label) / rotated_label.size > 0.001) and \
           (np.count_nonzero(rotated_label) / rotated_label.size < 0.98):
            valid_transform = True

        num_tries += 1

    return rotated_label

def random_cut(label, noise_params):
    """Randomly cut part of mask.

    Args:
        label: a [H, W] numpy array of {0, 1}
        noise_params: a Python dictionary.
    """

    H, W = label.shape

    num_tries = 0
    valid_transform = False
    while not valid_transform:

        if num_tries >= noise_params['max_augmentation_tries']:
            print('Cut: Exhausted number of augmentation tries...')
            return label

        cut_label = label.copy()

        # Sample cut percentage
        cut_percentage = np.random.uniform(noise_params['cut_percentage_min'],
                                           noise_params['cut_percentage_max'])

        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)
        if np.random.rand() < 0.5: # choose width
            
            sidelength = x_max - x_min
            if np.random.rand() < 0.5:  # from the left
                x = int(round(cut_percentage * sidelength)) + x_min
                cut_label[y_min:y_max+1, x_min:x] = 0
            else: # from the right
                x = x_max - int(round(cut_percentage * sidelength))
                cut_label[y_min:y_max+1, x:x_max+1] = 0

        else: # choose height
            
            sidelength = y_max - y_min
            if np.random.rand() < 0.5:  # from the top
                y = int(round(cut_percentage * sidelength)) + y_min
                cut_label[y_min:y, x_min:x_max+1] = 0
            else: # from the bottom
                y = y_max - int(round(cut_percentage * sidelength))
                cut_label[y:y_max+1, x_min:x_max+1] = 0

        # Make sure the mass is reasonable
        if (np.count_nonzero(cut_label) / cut_label.size > 0.001) and \
           (np.count_nonzero(cut_label) / cut_label.size < 0.98):
            valid_transform = True

        num_tries += 1

    return cut_label


def random_add(label, noise_params):
    """Randomly add part of mask .

    Args:
        label: a [H, W] numpy array of {0, 1}
        noise_params: a Python dictionary.
    """
    H, W = label.shape

    num_tries = 0
    valid_transform = False
    while not valid_transform:
        if num_tries >= noise_params['max_augmentation_tries']:
            print('Add: Exhausted number of augmentation tries...')
            return label

        added_label = label.copy()

        # Sample add percentage
        add_percentage = np.random.uniform(noise_params['add_percentage_min'],
                                           noise_params['add_percentage_max'])

        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)

        # Sample translation from center
        translation_percentage_x = np.random.uniform(0, 2*add_percentage)
        tx = int(round( (x_max - x_min) * translation_percentage_x ))
        translation_percentage_y = np.random.uniform(0, 2*add_percentage)
        ty = int(round( (y_max - y_min) * translation_percentage_y ))

        if np.random.rand() < 0.5: # choose x direction

            sidelength = x_max - x_min
            ty = np.random.choice([-1, 1]) * ty # mask will be moved to the left/right. up/down doesn't matter

            if np.random.rand() < 0.5: # mask copied from the left. 
                x = int(round(add_percentage * sidelength)) + x_min
                try:
                    temp = added_label[y_min+ty : y_max+1+ty, x_min-tx : x-tx]
                    added_label[y_min+ty : y_max+1+ty, x_min-tx : x-tx] = np.logical_or(temp, added_label[y_min : y_max+1, x_min : x])
                except ValueError as e: # indices were out of bounds
                    num_tries += 1
                    continue
            else: # mask copied from the right
                x = x_max - int(round(add_percentage * sidelength))
                try:
                    temp = added_label[y_min+ty : y_max+1+ty, x+tx : x_max+1+tx]
                    added_label[y_min+ty : y_max+1+ty, x+tx : x_max+1+tx] = np.logical_or(temp, added_label[y_min : y_max+1, x : x_max+1])
                except ValueError as e: # indices were out of bounds
                    num_tries += 1
                    continue

        else: # choose y direction

            sidelength = y_max - y_min
            tx = np.random.choice([-1, 1]) * tx # mask will be moved up/down. lef/right doesn't matter

            if np.random.rand() < 0.5:  # from the top
                y = int(round(add_percentage * sidelength)) + y_min
                try:
                    temp = added_label[y_min-ty : y-ty, x_min+tx : x_max+1+tx]
                    added_label[y_min-ty : y-ty, x_min+tx : x_max+1+tx] = np.logical_or(temp, added_label[y_min : y, x_min : x_max+1])
                except ValueError as e: # indices were out of bounds
                    num_tries += 1
                    continue
            else: # from the bottom
                y = y_max - int(round(add_percentage * sidelength))
                try:
                    temp = added_label[y+ty : y_max+1+ty, x_min+tx : x_max+1+tx]
                    added_label[y+ty : y_max+1+ty, x_min+tx : x_max+1+tx] = np.logical_or(temp, added_label[y : y_max+1, x_min : x_max+1])
                except ValueError as e: # indices were out of bounds
                    num_tries += 1
                    continue

        # Make sure the mass is reasonable
        if (np.count_nonzero(added_label) / added_label.size > 0.001) and \
           (np.count_nonzero(added_label) / added_label.size < 0.98):
            valid_transform = True

        num_tries += 1

    return added_label


