import sys, os
import json
import itertools
import cv2
from concurrent import futures
import functools
import collections

import torch
import torch.nn
import torch.nn.functional as F
import torch.distributions as tdist

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import cv2
from PIL import Image

from IPython.display import HTML as IP_HTML
from IPython.display import Image as IP_Image
from IPython.display import display as IP_display
import io


class AverageMeter(object):
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def torch_to_numpy(torch_tensor, is_standardized_image=False):
    """Convert torch tensor (NCHW) to numpy tensor (NHWC) for plotting.
    
    If it's an rgb image, it puts it back in [0,255] range (and undoes ImageNet standardization)

    Args:
        torch_tensor: a torch Tensor.

    Returns:
        a np.ndarray. 
    """
    np_tensor = copy_to_numpy(torch_tensor)
    if np_tensor.ndim == 4: # NCHW
        np_tensor = np_tensor.transpose(0,2,3,1)
    if is_standardized_image:
        _mean=[0.485, 0.456, 0.406]; _std=[0.229, 0.224, 0.225]
        for i in range(3):
            np_tensor[...,i] *= _std[i]
            np_tensor[...,i] += _mean[i]
        np_tensor *= 255
        np_tensor = np_tensor.astype(np.uint8)

    return np_tensor


def copy_to_numpy(m):
    """Copy tensor (either numpy array or torch tensor) to a numpy array."""
    if isinstance(m, np.ndarray):
        m = m.copy()
    elif torch.is_tensor(m):
        m = m.cpu().clone().detach().numpy()
    else:
        raise NotImplementedError("MUST pass torch tensor or numpy array")
    return m

        
def copy_to_torch(m, cuda=False):
    """Copy tensor (either numpy array or torch tensor) to a numpy array."""
    if torch.is_tensor(m):
        m = m.clone()
    elif isinstance(m, np.ndarray):
        m = torch.from_numpy(m)
    else:
        raise NotImplementedError("MUST pass torch tensor or numpy array")
    
    if cuda:
        m = m.cuda()
    else:
        m = m.cpu()
    
    return m


def normalize(M):
    """Normalize values of M to the range [0,1]."""
    M = M.astype(np.float32)
    return (M - M.min()) / (M.max() - M.min())
        

def get_color_mask(object_index, nc=None):
    """Convert segmentation image to color image.

    Colors each index differently. Useful for visualizing semantic masks.

    Args:
        object_index: a [H, W] numpy array of ints from {0, ..., nc-1}
        nc: int. total number of colors. If None, this will be inferred by masks

    Returns:
        a [H, W, 3] np.ndarray of type uint8.
    """
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255
        
    return color_mask


def build_matrix_of_indices(height, width):
    """Build a [height, width, 2] numpy array containing coordinates.

    Args:
        height: int.
        width: int.

    Returns:
        np.ndarray B [H, W, 2] s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)


def torch_moi(h, w, device='cpu'):
    """Build matrix of indices in pytorch.

    Torch function to do the same thing as build_matrix_of_indices, but returns CHW format.
        
    Args:
        h: int
        w: int

    Returns:
        torch.FloatTensor B [2, H, W] s.t. B[0, ...] contains y-coordinates, B[1, ...] contains x-coordinates
    """
    ys = torch.arange(h, device=device).view(-1,1).expand(h,w)
    xs = torch.arange(w, device=device).view(1,-1).expand(h,w)
    return torch.stack([ys, xs], dim=0).float()


def consecutive_label_img(labels):
    """ Map labels to {0, 1, ..., K-1}. 

    Args:
        labels: a [H, W] np.ndarray with integer values

    Returns:
        a [H, W] np.ndarray
    """

    # Find the unique (nonnegative) labels, map them to {0, ..., K-1}
    unique_nonnegative_indices = np.unique(labels)
    mapped_labels = labels.copy()
    for k in range(unique_nonnegative_indices.shape[0]):
        mapped_labels[labels == unique_nonnegative_indices[k]] = k
    return mapped_labels


def visualize_segmentation(im, masks, nc=None, save_dir=None):
    """Visualize segmentations nicely. 

    Based on code from:
        https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

    Args:
        im: a [H, W, 3] RGB image. numpy array of dtype np.uint8
        masks: a [H, W] numpy array of dtype np.uint8 with values in {0, ..., K}
        nc: int. total number of colors. If None, this will be inferred by masks

    Returns:
        A [H, W, 3] RGB image as a numpy array. 
            OR
        PIL Image instance.
    """ 
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    masks = consecutive_label_img(masks)
    im = im.copy()

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    # Mask
    imgMask = np.zeros(im.shape)


    # Draw color masks
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)


    # Draw mask contours
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Find contours
        contour, hier = cv2.findContours(
            e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Plot the nice outline
        for c in contour:
            cv2.drawContours(im, contour, -1, (255,255,255), 2)


    if save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image
    else:
        return im


def visualize_contour_img(contour_mean, contour_std, rgb_img):
    """Visualize uncertainty estimates from RICE.

    Args:
        contour_mean: a [H, W] np.ndarray with values in [0,1].
        contour_std: a [H, W] np.ndarray with values in [0, inf).
        rgb_img: a [H, W, 3] np.ndarray.

    Returns:
        a [H, W, 3] np.ndarray.
    """
    
    image_H, image_W = rgb_img.shape[:2]
    contour_img = np.round(contour_mean * 255).astype(np.uint8)
    contour_img = np.stack([np.zeros((image_H, image_W), dtype=np.uint8),
                            contour_img,
                            np.zeros((image_H, image_W), dtype=np.uint8)], axis=-1)
    contour_std_img = np.round(normalize(contour_std) * 255).astype(np.uint8)
    contour_std_img = np.stack([contour_std_img,
                                np.zeros((image_H, image_W), dtype=np.uint8),
                                np.zeros((image_H, image_W), dtype=np.uint8)], axis=-1)
    contour_img[contour_std_img[...,0] > 0] = 0
    contour_img[contour_std_img > 0] = contour_std_img[contour_std_img > 0]

    contour_img = cv2.addWeighted(rgb_img, 0.25, contour_img, 0.75, 0.0)
    
    return contour_img
    

### These two functions were adatped from the DAVIS public dataset ###

def imread_indexed(filename):
    """Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation


def mask_to_tight_box_numpy(mask):
    """Return bbox given mask.

    Args:
        mask: a [H, W] numpy array

    Returns:
        a 4-tuple of scalars.
    """
    a = np.transpose(np.nonzero(mask))
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max


def mask_to_tight_box_pytorch(mask):
    """Return bbox given mask.

    Args:
        mask: a [H, W] torch tensor

    Returns:
        a 4-tuple of torch scalars.
    """
    a = torch.nonzero(mask)
    bbox = torch.min(a[:, 1]), torch.min(a[:, 0]), torch.max(a[:, 1]), torch.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max


def mask_to_tight_box(mask):
    if type(mask) == torch.Tensor:
        return mask_to_tight_box_pytorch(mask)
    elif type(mask) == np.ndarray:
        return mask_to_tight_box_numpy(mask)
    else:
        raise Exception(f"Data type {type(mask)} not understood for mask_to_tight_box...")


def compute_xyz(depth_img, camera_params):
    """Compute ordered point cloud from depth image and camera parameters.
        
    Assumes camera uses left-handed coordinate system, with 
        x-axis pointing right
        y-axis pointing up
        z-axis pointing "forward"

    Args:
        depth_img: a [H, W] numpy array of depth values in meters
        camera_params: a dictionary with camera parameters

    Returns:
        a [H, W, 3] numpy array
    """

    # Compute focal length from camera parameters
    if 'fx' in camera_params and 'fy' in camera_params:
        fx = camera_params['fx']
        fy = camera_params['fy']
    else: # simulated data
        aspect_ratio = camera_params['img_width'] / camera_params['img_height']
        e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
        t = camera_params['near'] / e; b = -t
        r = t * aspect_ratio; l = -r
        alpha = camera_params['img_width'] / (r-l)  # pixels per meter
        focal_length = camera_params['near'] * alpha  # focal length of virtual camera (frustum camera)
        fx = focal_length; fy = focal_length

    if 'x_offset' in camera_params and 'y_offset' in camera_params:
        x_offset = camera_params['x_offset']
        y_offset = camera_params['y_offset']
    else: # simulated data
        x_offset = camera_params['img_width']/2
        y_offset = camera_params['img_height']/2

    indices = build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0])  # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # [H, W, 3]

    return xyz_img

def unique_useable_mask_labels(masks, to_ignore=[0]):

    if type(masks) == torch.Tensor:
        mask_labels = torch.unique(masks).float()
    elif type(masks) == np.ndarray:
        mask_labels = np.unique(masks).astype(np.float32)
    else:
        raise Exception(f"Data type {type(masks)} not understood for convert_mask_HW_to_NHW...")

    useable = mask_labels > -1 # all True
    for ig in to_ignore:
        useable = useable & (mask_labels != ig)
    mask_labels = mask_labels[useable]

    return mask_labels


def convert_mask_HW_to_NHW_pytorch(masks, to_ignore=[0], to_keep=[]):
    """Convert HW format to NHW format.

    Convert masks of shape [H, W] with values in {2, 3, ..., N+1} to
        masks of shape [N, H, W] with values in {0,1}.

    Args:
        masks: torch.FloatTensor of shape [H, W]

    Returns:
        torch.FloatTensor of shape [N, H, W]
    """

    H, W = masks.shape

    mask_labels = unique_useable_mask_labels(masks, to_ignore=to_ignore)
    if len(to_keep) > 0:
        temp = set(mask_labels.cpu().numpy()).union(set(to_keep))
        mask_labels = torch.tensor(sorted(list(temp))).to(masks.device)
    K = mask_labels.shape[0]

    new_masks = torch.zeros((K,H,W), dtype=torch.float32, device=masks.device)
    for k, label in enumerate(mask_labels):
        new_masks[k] = (masks == label).float()

    return new_masks


def convert_mask_HW_to_NHW_numpy(masks, to_ignore=[0], to_keep=[]):
    """Convert HW format to NHW format.

    Convert masks of shape [H, W] with values in {2, 3, ..., N+1} to
        masks of shape [N, H, W] with values in {0,1}.

    Args:
        masks: np.ndarray of shape [H, W]

    Returns:
        np.ndarray of shape [N, H, W]
    """

    H, W = masks.shape

    mask_labels = unique_useable_mask_labels(masks, to_ignore=to_ignore)
    if len(to_keep) > 0:
        temp = set(mask_labels).union(set(to_keep))
        mask_labels = np.array(sorted(list(temp)))
    K = mask_labels.shape[0]

    new_masks = np.zeros((K,H,W), dtype=masks.dtype)
    for k, label in enumerate(mask_labels):
        new_masks[k] = (masks == label).astype(masks.dtype)

    return new_masks


def convert_mask_HW_to_NHW(masks, to_ignore=[0], to_keep=[]):
    """Convert HW format to NHW format."""
    if type(masks) == torch.Tensor:
        return convert_mask_HW_to_NHW_pytorch(masks, to_ignore=to_ignore, to_keep=to_keep)
    elif type(masks) == np.ndarray:
        return convert_mask_HW_to_NHW_numpy(masks, to_ignore=to_ignore, to_keep=to_keep)
    else:
        raise Exception(f"Data type {type(masks)} not understood for convert_mask_HW_to_NHW...")


def convert_mask_NHW_to_HW_pytorch(masks, start_label=1):
    """Convert NHW format to HW format.

    Convert masks of shape [N, H, W] with values in {0,1} to
        masks of shape [H, W] with values in {2, 3, ..., N+1}.

    Args:
        masks: torch.FloatTensor of shape [N, H, W]

    Returns:
        torch.FloatTensor of shape [H, W]
    """

    N = masks.shape[0]
    temp = torch.arange(start_label, N+start_label, device=masks.device)[:,None,None] * masks
    return torch.sum(temp, dim=0) # Shape: [H, W]


def convert_mask_NHW_to_HW_numpy(masks, start_label=1):
    """Convert NHW format to HW format.

    Convert masks of shape [N, H, W] with values in {0,1} to
        masks of shape [H, W] with values in {2, 3, ..., N+1}.

    Args:
        masks: np.ndarray of shape [N, H, W]

    Returns:
        np.ndarray of shape [H, W]
    """

    N = masks.shape[0]
    temp = np.arange(start_label, N+start_label)[:,None,None] * masks
    return np.sum(temp, axis=0)  # [H, W]


def convert_mask_NHW_to_HW(masks, start_label=1):
    """Convert NHW format to HW format."""
    if type(masks) == torch.Tensor:
        return convert_mask_NHW_to_HW_pytorch(masks, start_label=start_label)
    elif type(masks) == np.ndarray:
        return convert_mask_NHW_to_HW_numpy(masks, start_label=start_label)
    else:
        raise Exception(f"Data type {type(masks)} not understood for convert_mask_NHW_to_HW...")


def dilate(mask, size=3):
    """Dilation operation in Pytorch.

    Args:
        mask: a [N, H, W] torch.FloatTensor with values in {0,1}
        size: a odd integer describing dilation kernel radius

    Returns:
        a [N, H, W] torch.FloatTensor with values in {0,1}
    """
    assert size % 2 == 1 # size MUST be odd

    mask = mask.unsqueeze(0) # Shape: [1, N, H, W]
    dilated_mask = F.max_pool2d(mask, size, stride=1, padding=size//2)

    return dilated_mask[0]


def is_neighboring_mask(mask1, mask2, size=5):
    """Compute if mask1 is touching mask2.

    Args:
        mask1: a [N, H, W] torch.FloatTensor with values in {0,1}
        mask2: a [N, H, W] torch.FloatTensor with values in {0,1}
        size: size of dilation kernel to determine "neighbors"
                size // 2 = #pixels to dilate

    Returns:
        a [N] torch.BoolTensor
    """
    d_mask1 = dilate(mask1, size=size)
    return (d_mask1 * mask2 > 0).any(dim=2).any(dim=1)


def neighboring_mask_indices(masks, neighbor_dist=50, batch_size=50, reduction_factor=1):
    """Return pairs of mask indices that are neighboring.

    Args:
        masks: a [N, H, W] torch.FloatTensor
        neighbor_dist: a Python int. Max distance of masks to be considered as "neighboring"
                       Note: neighbor_dist needs to be >= reduction_factor
        batch_size: int.
        reduction_factor: int.

    Returns:
        a [n, 2] torch.LongTensor of indices, where n = |neighboring masks pairs| <= N*(N-1)/2
    """
    assert neighbor_dist >= reduction_factor, "<neighbor_dist> MUST be >= <reduction_factor>"

    N, H, W = masks.shape
    if N == 1:
        return torch.zeros((0,2), dtype=torch.long, device=masks.device)
    resized_masks = F.interpolate(masks.unsqueeze(0), size = (H//reduction_factor,W//reduction_factor), 
                                mode='nearest')[0] # Subsample by factor of 4

    indices = torch.tensor(list(itertools.combinations(range(N), 2)), device=masks.device) # Shape: [N*(N-1)/2, 2]
    N_pairs = indices.shape[0]

    neighboring = torch.zeros(N_pairs, dtype=torch.bool, device=masks.device)
    for i in range(0, N_pairs, batch_size):
        neighboring[i:i+batch_size] = is_neighboring_mask(resized_masks[indices[i:i+batch_size,0]],
                                                          resized_masks[indices[i:i+batch_size,1]],
                                                          size=neighbor_dist//reduction_factor*2+1)

    return indices[neighboring, :]


def graph_eq(graph_1, graph_2):
    """Graph equivalence.

    Return True if the original masks are exactly the same.
    """
    return ((graph_1.orig_masks.shape == graph_2.orig_masks.shape) and
            torch.all(graph_1.orig_masks == graph_2.orig_masks))


def mask_corresponding_gt(masks, gt_labels):
    """Find corresponding GT label for each mask.

    Args:
        masks: a [N, H, W] torch.FloatTensor of values in {0,1}. N = #objects/nodes
        gt_labels: a [N_gt, H, W] torch.FloatTensor of values in {0,1}. N_gt = #GT objects

    Returns:
        a [N] torch.LongTensor of values in {0, 1, ..., N_gt}
    """

    N, H, W = masks.shape
    N_gt = gt_labels.shape[0]

    mask_labels = torch.zeros((N,), dtype=torch.long, device=masks.device)
    if N_gt == 0:
        return mask_labels

    batch_size_N = max(10 // N_gt, 1)
    for i in range(0, N, batch_size_N):
        intersection = masks[i:i+batch_size_N].unsqueeze(1).long() & gt_labels.unsqueeze(0).long() # Shape: [batch_size_N, N_gt, H, W]
        mask_labels[i:i+batch_size_N] = torch.argmax(intersection.sum(dim=(2,3)), dim=1)

    return mask_labels


def crop_indices_with_padding(mask, config, inference=False):
    """Randomly pad mask, crop it.

    Args:
        mask: a [H, W] torch.FloatTensor with values in {0,1}
        config: a Python dictionary with keys:
                       - padding_alpha
                       - padding_beta
                       - min_padding_percentage

    Returns:
        x_min, y_min, x_max, y_max
    """
    H, W = mask.shape
    x_min, y_min, x_max, y_max = mask_to_tight_box(mask)

    # Make bbox square
    x_delta = x_max - x_min
    y_delta = y_max - y_min
    if x_delta > y_delta:
        y_max = y_min + x_delta
    else:
        x_max = x_min + y_delta

    sidelength = x_max - x_min
    
    if inference:
        x_padding = int(torch.round((x_max - x_min).float() * config['padding_percentage']).item())
        y_padding = int(torch.round((y_max - y_min).float() * config['padding_percentage']).item())

    else:
        padding_percentage = tdist.Beta(config['padding_alpha'], config['padding_beta']).sample()
        padding_percentage = max(padding_percentage, config['min_padding_percentage'])
        padding = int(torch.round(sidelength * padding_percentage).item())
        if padding == 0:
            print(f'Whoa, padding is 0... sidelength: {sidelength}, %: {padding_percentage}')
            padding = 25 # just make it 25 pixels
        x_padding = padding
        y_padding = padding


    # Pad and be careful of boundaries
    x_min = (x_min - x_padding).clamp(min=0)
    x_max = (x_max + x_padding).clamp(max=W-1)
    y_min = (y_min - y_padding).clamp(min=0)
    y_max = (y_max + y_padding).clamp(max=H-1)

    # if (y_min == y_max) or (x_min == x_max):
    #     print('Something is wrong with crop indices:', x_min, y_min, x_max, y_max)

    return x_min, y_min, x_max, y_max


def seg2bmap(seg, return_contour=False):
    """Compute boundary map from segmentation.

     From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries. This boundary lives on the mask, i.e. it's a subset of the mask.

    Args:
        seg: a [H, W] numpy array of values in {0,1}

    Returns:
        a [H, W] numpy array of values in {0,1}
        a [2, num_boundary_pixels] numpy array. [0,:] is y-indices, [1,:] is x-indices
    """
    seg = seg.astype(np.uint8)
    contours, hierarchy = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    temp = np.zeros_like(seg)
    bmap = cv2.drawContours(temp, contours, -1, 1, 1)

    if return_contour: # Return the SINGLE largest contour
        contour_sizes = [len(c) for c in contours]
        ind = np.argmax(contour_sizes)
        contour = np.ascontiguousarray(np.fliplr(contours[ind][:,0,:]).T) # Shape: [2, num_boundary_pixels]
        return bmap, contour
    else:
        return bmap


def mask_boundary_overlap(m0, m1, d=2):
    """Compute overlap between mask boundaries.

    Args:
        m0: a [H, W] numpy array of values in {0,1}
        m1: a [H, W] numpy array of values in {0,1}
        d: dilation factor

    Returns:
        a [H, W] numpy array of values in {0,1}
    """

    # Compute boundaries
    temp0 = seg2bmap(m0)
    temp1 = seg2bmap(m1)

    # Dilate boundaries, AND them with other boundary
    temp0_d1 = temp0 & cv2.dilate(temp1, np.ones((d,d), dtype=np.uint8), iterations=1)
    temp1_d0 = temp1 & cv2.dilate(temp0, np.ones((d,d), dtype=np.uint8), iterations=1)

    return temp0_d1 | temp1_d0
    

def sigmoid(x):
    return 1/(1+np.exp(-x))


def largest_connected_component(mask, connectivity=4):
    """Run connected components algorithm and return mask of largest one.

    Args:
        mask: a [H, W] numpy array 

    Returns:
        a [H, W] numpy array of same type as input
    """

    # Run connected components algorithm
    num_components, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)

    # Find largest connected component via set distance
    largest_component_num = -1
    largest_component_size = -1 
    for j in range(1, num_components):
        component_size = np.count_nonzero(components == j)
        if component_size > largest_component_size:
            largest_component_num = j
            largest_component_size = component_size

    return (components == largest_component_num).astype(mask.dtype)


def filter_out_empty_masks_NHW(masks, min_pixels_thresh=1., start_label=1):
    """Filter out empty masks.

    Args:
        masks: a [N, H, W] torch.FloatTensor with values in {0,1}

    Returns:
        a [N_filter, H, W] torch.FloatTensor, where N_filter = number of masks after filtering.
    """
    shape_HW = masks.ndim == 2
    if shape_HW:
        masks = convert_mask_HW_to_NHW(masks)
    keep_inds = masks.sum(dim=(1,2)) >= min_pixels_thresh
    masks = masks[keep_inds]
    if shape_HW:
        return convert_mask_NHW_to_HW(masks, start_label=start_label)
    else:
        return masks


def subplotter(images, suptitle=None, max_plots_per_row=4, fig_index_start=1, **kwargs):
    """Plot images side by side.
    
    Args:
        images: an Iterable of [H, W, C] np.arrays. If images is
            a dictionary, the values are assumed to be the arrays,
            and the keys are strings which will be titles.
    """
    
    if type(images) not in [list, dict]:
        raise Exception("images MUST be type list or dict...")

    fig_index = fig_index_start
    
    num_plots = len(images)
    num_rows = int(np.ceil(num_plots / max_plots_per_row))

    for row in range(num_rows):

        fig = plt.figure(fig_index, figsize=(max_plots_per_row*5, 5))
        fig_index += 1

        if row == 0 and suptitle is not None:
            fig.suptitle(suptitle)

        for j in range(max_plots_per_row):

            ind = row*max_plots_per_row + j
            if ind >= num_plots:
                break

            plt.subplot(1, max_plots_per_row, j+1)
            if type(images) == dict:
                title = list(images.keys())[ind]
                image = images[title]
                plt.title(title)
            else:
                image = images[ind]
            plt.imshow(image, **kwargs)


def gallery(images, width='auto'):
    """Shows a set of images in a gallery that flexes with the width of the notebook.
    
    Args:
        images: an Iterable of [H, W, C] np.arrays. If images is
            a dictionary, the values are assumed to be the arrays,
            and the keys are strings which will be titles.

        width: str
            CSS height value to assign to all images. Set to 'auto' by default to show images
            with their native dimensions. Set to a value like '250px' to make all rows
            in the gallery equal height.
    """
    def _src_from_data(data):
        """Base64 encodes image bytes for inclusion in an HTML img element"""
        img_obj = IP_Image(data=data)
        for bundle in img_obj._repr_mimebundle_():
            for mimetype, b64value in bundle.items():
                if mimetype.startswith('image/'):
                    return f'data:{mimetype};base64,{b64value}'

    def _get_img_as_bytestring(img):
        im = Image.fromarray(img)
        buf = io.BytesIO()
        im.save(buf, format='JPEG')
        return buf.getvalue()
    
    if not (isinstance(images, list) or isinstance(images, dict)):
        raise Exception("images MUST be type list or dict...")
    
    num_images = len(images)
    
    figures = []
    for i in range(num_images):
        if isinstance(images, list):
            caption = ''
            image = images[i]
        else: # dict
            caption = list(images.keys())[i]
            image = images[caption]
        src = _src_from_data(_get_img_as_bytestring(image))

        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="width: {width}">
              {caption}
            </figure>
        ''')
        
    IP_display(IP_HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    '''))


def parallel_map(f, *args, **kwargs):
    """Parallel version of map().
    
    Args:
        f: function handle.
        *args: Every element of args (list) MUST be an iterable. The iterables 
            must have the same length.
        **kwargs: keyword dictionary.
        
    Returns:
        a list of outputs (of f)
    """
    partial_f = functools.partial(f, **kwargs) if kwargs else f
    with futures.ThreadPoolExecutor() as executor:
        results = executor.map(partial_f, *args)
        return list(results)
    

def parallel_map_dict(f, dict_, **kwargs):
    """Apply f to each element of dict_.values()."""
    ordered_dict = collections.OrderedDict(dict_)
    results = parallel_map(f, ordered_dict.values(), **kwargs)
    return {key: results[i] for i, key in enumerate(ordered_dict.keys())}


def load_uoisnet_3d(cfg_filename,
                    dsn_filename,
                    rrn_filename):
    """Load UOIS-Net-3D."""
    from ..uois.src.segmentation import UOISNet3D
    import yaml
    
    with open(cfg_filename, 'r') as f:
        uoisnet_3d_config = yaml.load(f)

    return UOISNet3D(uoisnet_3d_config['uois_config'], 
                     dsn_filename,
                     uoisnet_3d_config['dsn_config'],
                     rrn_filename,
                     uoisnet_3d_config['rrn_config'])

