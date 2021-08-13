import numpy as np
import cv2
import torch

from . import data_augmentation
from . import graph_construction as gc
from .util import utilities as util_


def _compute_mask_centers(masks):
    """Compute mask centers.

    Args:
        masks: a [N, H, W] torch.Tensor with values in {0, 1}.

    Returns:
        a [N, 2] np.ndarray of (x, y) centers for each mask.
    """
    H, W = masks.shape[1:3]
    pixel_indices = util_.build_matrix_of_indices(H, W)

    N = masks.shape[0]
    centers = np.zeros((N, 2), dtype=int)
    for i in range(N):
        object_mask = masks[i].bool().cpu().numpy()
        center = np.mean(pixel_indices[object_mask, :], axis=0)  # [2]. y_center, x_center
        center = center.round().astype(int)[::-1]  # [2]. x_center, y_center
        centers[i] = center

    return centers


def _draw_nodes_and_edges(graph, graph_img, centers):

    for i in range(graph.orig_masks.shape[0]):
        cv2.circle(graph_img, tuple(centers[i]), 12, (255,0,0), -1)

    if 'edge_index' in graph:
        for pair in graph.edge_index.T:
            src = pair[0]
            dst = pair[1]
            cv2.line(graph_img, tuple(centers[src]), tuple(centers[dst]), (0,0,255), 2)


def visualize_graph(rgb_img, graph, mode='side_by_side'):
    """Visualize graph segmentation.

    Args:
        rgb_img: a [3, H, W] torch tensor OR a [H, W, 3] np.ndarray. Can be standardized or not.
        graph: a torch_geometric.Data instance
        mode: string. MUST be in ['side_by_side', 'graph_on_rgb', 'seg_graph_on_rgb']
    """
    if rgb_img.ndim != 3:
        raise NotImplementedError(f"RGB image MUST have rank 3. [H, W, 3] or [3, H, W]")

    if torch.is_tensor(rgb_img):  # assume this is in [3 X H x W]
        rgb_img = rgb_img.permute(1,2,0).cpu().numpy()
    if rgb_img.min() < 0:  # assume this means standardized
        rgb_img = data_augmentation.unstandardize_image(rgb_img)
    H, W = rgb_img.shape[:2]

    # Compute graph image
    graph = gc.remove_bg_node(graph)
    graph_img = np.zeros_like(rgb_img)

    centers = _compute_mask_centers(graph.orig_masks)
    _draw_nodes_and_edges(graph, graph_img, centers)

    # Plot image
    if mode == 'side_by_side':
        # Paint RGB and graph side by side
        final_img = np.zeros((H, W*2, 3), dtype=np.uint8)
        final_img[:, 0:W, :] = rgb_img
        final_img[:, W:W*2, :] = graph_img

    elif mode == 'graph_on_rgb':
        # Paint graph onto RGB
        dimming_param = .75 # 1 means no dim, 0 means completely blackout
        final_img = (rgb_img * dimming_param).round().astype(np.uint8)
        graph_mask = (graph_img>0).max(axis=2)
        final_img[graph_mask,:] = graph_img[graph_mask,:]

    elif mode == 'seg_graph_on_rgb':
        dimming_param = .8 # 1 means no dim, 0 means completely blackout
        masks = util_.convert_mask_NHW_to_HW(graph.orig_masks)
        seg_on_image = util_.visualize_segmentation(rgb_img, masks.cpu().numpy().astype(np.uint8))
        final_img = (seg_on_image * dimming_param).round().astype(np.uint8)
        graph_mask = (graph_img>0).max(axis=2)
        final_img[graph_mask,:] = graph_img[graph_mask,:]

    else:
        raise NotImplementedError("'mode' MUST be in ['side_by_side', 'graph_on_rgb', 'seg_graph_on_rgb']...")

    return final_img

