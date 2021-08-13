
import itertools
from collections import OrderedDict
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

import skimage
import skimage.graph

from . import base_networks
from . import graph_construction as gc
from . import constants
from .util import utilities as util_


def split_mask_upsample(orig_mask, split_masks, crop_indices):
    """Upsample split masks to original size.

    Args:
        orig_mask: a [H, W] torch.FloatTensor of values in {0, 1}
        split_masks: a [H', W'] torch.FloatTensor of values in {0, 1, 2, ...}
        crop_indices: a [4] torch.FloatTensor

    Returns:
        a [H, W] torch.FloatTensor of values in {0, 1, 2, ...}
    """
    x_min, y_min, x_max, y_max = crop_indices
    h, w = y_max-y_min+1, x_max-x_min+1
    ys = slice(y_min, y_max+1)
    xs = slice(x_min, x_max+1)

    split_masks = F.interpolate(split_masks[None,None,...], (h, w), mode='nearest')[0,0]
    temp = torch.zeros_like(orig_mask)  # [H, W]
    temp[ys, xs] = split_masks
    split_masks = temp
    upsampled_split_masks = torch.zeros_like(orig_mask)

    # Copy intersection over
    orig_mask_bool = orig_mask > 0.5
    split_masks_bool = split_masks > 0.5  # Recall: split_masks can have mask IDs of 1, 2, ...
    intersection = orig_mask_bool & split_masks_bool
    upsampled_split_masks[intersection] = split_masks[intersection]

    # Look at non-filled pixels, find nearest neighbor in split mask
    non_filled = orig_mask_bool & ~split_masks_bool
    non_filled_idxs = torch.stack(torch.where(non_filled)).T  # [#non_filled, 2]
    if non_filled_idxs.shape[0] > 0:
        non_filled_dilated = util_.dilate(non_filled.float(), size=9).bool()  # Only look at neighbors, not entire mask. That wastes GPU memory
        pixels_to_consider = split_masks_bool & non_filled_dilated
        pixels_to_consider = torch.stack(torch.where(pixels_to_consider)).T  # [#split_mask, 2]
        distances = torch.norm(non_filled_idxs.float().unsqueeze(1) - pixels_to_consider.float().unsqueeze(0), dim=2)  # [#non_filled, #split_mask]
        nf_nn_indices = torch.argmin(distances, dim=1)  # [#non_filled]

        # Fill in with nearest neighbor
        upsampled_split_masks[non_filled_idxs[:,0],
                              non_filled_idxs[:,1]] = split_masks[pixels_to_consider[nf_nn_indices][:,0],
                                                                  pixels_to_consider[nf_nn_indices][:,1]]

    return upsampled_split_masks


class SplitNet(nn.Module):

    def __init__(self, config):
        super(SplitNet, self).__init__()
        self.node_encoder = base_networks.NodeEncoder(config['node_encoder_config'])
        self.decoder = base_networks.SplitNetDecoder(config['decoder_config'])

    def forward(self, graph):
        """Forward pass of SplitNet.

        Note: Do not split the background node

        Args:
            graph: a torch_geometric.Data (Batch) instance with keys:
                     - rgb: a [N, 256, h, w] torch.FloatTensor of ResnNet50+FPN rgb image features
                     - depth: a [N, 3, h, w] torch.FloatTensor. XYZ image
                     - mask: a [N, h, w] torch.FloatTensor of values in {0, 1}
                     - orig_masks: a [N, H, W] torch.FloatTensor of values in {0, 1}. Original image size.
                     - crop_indices: a [N, 4] torch.LongTensor. xmin, ymin, xmax, ymax.

        Returns:
            a [N] torch.FloatTensor of split score logits
            a [N, h, w] torch.FloatTensor of boundary score logits
        """
        graph = gc.remove_bg_node(graph)
        temp = self.decoder(self.node_encoder(graph))
        split_score_logits = temp[0][:,0]  # [N-1]
        boundary_logits = temp[1][:,0]  # [N-1, h, w]

        h, w = boundary_logits.shape[1:3]

        # Handle background
        split_score_logits = torch.cat([torch.ones((1), device=constants.DEVICE) * -100,
                                        split_score_logits], dim=0)
        boundary_logits = torch.cat([torch.ones((1, h, w), device=constants.DEVICE) * -100,
                                     boundary_logits], dim=0)

        return split_score_logits, boundary_logits


class SplitNetWrapper(base_networks.NetworkWrapper):

    def setup(self):

        self.model = SplitNet(self.config)
        self.model.to(self.device)


    def split_mask(self, mask, split_boundary, num_start_end_samples=5, return_path_cost=False):
        """Sample a split of the mask.

        Args:
            mask: a [H', W'] torch.FloatTensor with values in {0,1}
            split_boundary: a [H', W'] torch.FloatTensor of boundary scores
                            Note: mask and split_boundary have already been resized

        Returns:
            a [H, W] torch.FloatTensor with values in {0, 1, ...}
        """
        def retval(mask=mask, path_cost=None, path=None):
            if return_path_cost:
                return mask, path_cost, path
            else:
                return mask

        new_H, new_W = split_boundary.shape

        # Compute boundary and contour topology
        mask_boundary, contour = util_.seg2bmap(mask.cpu().numpy(), return_contour=True)
        contour = torch.from_numpy(contour).to(self.device).long() # Shape: [2, num_boundary_pixels]
        num_boundary_pixels = contour.shape[1]

        # Compute component sizes of thresholded boundary probabilities
        bp_thresholded = (split_boundary > self.config['boundary_probability_threshold']).cpu().numpy()
        num_components, components = cv2.connectedComponents(bp_thresholded.astype(np.uint8), connectivity=8)
        component_size_img = np.zeros(bp_thresholded.shape, dtype=np.float32)
        for j in range(1, num_components):
            component_size = np.count_nonzero(components == j)
            component_size_img[components == j] = component_size
        component_size_img = torch.from_numpy(component_size_img).to(self.device) # ['H, W']

        # Compute distance filters with Gaussian weights
        sigma = 1
        moi = util_.build_matrix_of_indices(new_H, new_W) # Shape: [H', W', 2]
        moi = torch.from_numpy(moi).to(self.device)
        distance_filters = torch.norm(moi[...,None] - contour, dim=2) # Shape: [H, W, num_boundary_pixels]
        distance_filters = torch.exp( - .5 * (distance_filters/sigma)**2) 
        normalized_distance_filters = distance_filters / torch.sum(distance_filters, dim=(0,1), keepdims=True) # Shape: [H, W, num_boundary_pixels]

        mask_boundary_probs = torch.sum(normalized_distance_filters * component_size_img[...,None], dim=(0,1)) # Shape: [num_boundary_pixels]
        if torch.allclose(mask_boundary_probs.sum(), torch.zeros(1).to(self.device)): # No predicted boundaries. move on
            # print("No mask boundary probabilities for split...")
            return retval(mask=mask)
        mask_boundary_probs = mask_boundary_probs / mask_boundary_probs.sum()




        #### SAMPLE START/END X TIMES AND SELECT LOWEST COST ####
        contour_closeness = (distance_filters.max(dim=2)[0] > 0.5).cpu().numpy() # Pre-compute a weighting that favors path being away from boundary
        best_path_cost = 1. # highest possible cost
        best_path = None
        components = torch.zeros_like(mask)
        for i in range(num_start_end_samples):

            # Sample start point
            start_index = torch.multinomial(mask_boundary_probs,1)
            start_px = contour[:,start_index.item()]

            # Sample end point. First, weight mask_boundary_probs with topological distance from start point
            sig_x_extreme = 6
            halfway = int(num_boundary_pixels/2)
            dist_from_st = np.arange(num_boundary_pixels).astype(np.float32)
            dist_from_st[halfway:] = util_.sigmoid(np.linspace(-sig_x_extreme, sig_x_extreme, num_boundary_pixels-halfway))
            dist_from_st[:halfway] = util_.sigmoid(np.linspace(sig_x_extreme, -sig_x_extreme, halfway))
            dist_from_st = np.roll(dist_from_st, start_index.item()-halfway)
            dist_from_st = torch.from_numpy(dist_from_st).to(self.device) # Shape: [num_boundary_pixels]

            # Multiply distance from start with mask boundary probabilities
            end_point_dist = dist_from_st * mask_boundary_probs # .5*dist_from_st + .5*mask_boundary_probs
            end_point_dist = end_point_dist / end_point_dist.sum() # Shape: [num_boundary_pixels]

            end_index = torch.argmax(end_point_dist)
            end_px = contour[:,end_index.item()]

            # Compute path from start to end
            boundary_cost = .7 # Cost to travel along boundary
            cost_img = 1-split_boundary.cpu().numpy()
            cost_img[contour_closeness] = boundary_cost

            path, _ = skimage.graph.route_through_array(cost_img, 
                                                        start_px.cpu().numpy(), 
                                                        end_px.cpu().numpy())
            path = np.array(path) # Shape: [path_length, 2]

            # Compute cost of path
            path_cost = cost_img[path[1:-1,0], path[1:-1,1]].mean() # Don't count the start/end, those are on the contour
            if path_cost >= best_path_cost or path.shape[0] <= 2: # path_length <= 2 means no path. just start/end, or start = end
                continue
            # The code below will execute only if it's better. The last time it's executed was the best
            best_path_cost = path_cost
            best_path = path

            # Compute actual split by using connected components after 0-ing out path
            disconnected_mask = mask.cpu().numpy().astype(np.uint8)
            disconnected_mask[path[:,0], path[:,1]] = 0
            num_components, components = cv2.connectedComponents(disconnected_mask, connectivity=4)

            # Figure out which component path belongs to
            best_component_num = -1
            best_component_overlap = -1 
            for j in range(1, num_components):
                component_size = np.count_nonzero(components == j)
                if component_size <= 5: # Set small components things to 0
                    components[components == j] = 0
                temp = cv2.dilate((components == j).astype(np.uint8), np.ones((3,3), dtype=np.uint8), iterations=1)
                component_overlap = np.sum(temp[path[:,0], path[:,1]])
                if component_overlap > best_component_overlap:
                    best_component_num = j
                    best_component_overlap = component_overlap
            components[path[:,0], path[:,1]] = best_component_num
            components = torch.from_numpy(components).float().to(self.device) # Shape: [H', W']

        ##### END X TIME LOOP #####

        return retval(mask=components, path_cost=best_path_cost, path=best_path)


    def split_scores_via_sampling_splits(self, graph):
        """Score the split-ability of the mask.
           
        Simply sample some split boundaries and choose the best one.
        """

        # Apply the model
        split_score_logits, boundary_logits = self.model(graph)
        boundary_probs = torch.sigmoid(boundary_logits)  # [N, h, w]
        boundary_probs = boundary_probs * graph.mask[:,0]  # Intersect it with mask. [N, h, w]

        graph.paths = dict()
        graph.split = torch.zeros_like(boundary_probs)
        split_scores = torch.zeros(boundary_probs.shape[0], device=self.device)
        for k in range(boundary_probs.shape[0]):
            if torch.sum(boundary_probs[k]) <= 0.01:
                continue

            else:
                temp, path_cost, path = self.split_mask(graph.mask[k,0], boundary_probs[k], 
                                                        num_start_end_samples=5, return_path_cost=True)
                if path_cost is not None: # else it stays as 0
                    split_scores[k] = 1-path_cost
                    graph.split[k] = temp
                    graph.paths[k] = path

        return split_scores


class MergeBySplitWrapper(base_networks.NetworkWrapper):

    def setup(self):

        if 'splitnet_model' in self.config:
            self.model = self.config['splitnet_model']
        else:
            self.model = SplitNet(self.config)
        self.model.to(self.device)


    def merge_scores_via_splitnet(self, graph, other_args):
        """Use SplitNet to compute merge scores

        Args:
            graph: a torch_geometric.Data (Batch) instance with keys:
                     - rgb: a [N, 256, h, w] torch.FloatTensor of ResnNet50+FPN rgb image features
                     - depth: a [N, 3, h, w] torch.FloatTensor. XYZ image
                     - mask: a [N, h, w] torch.FloatTensor of values in {0, 1}
                     - orig_masks: a [N, H, W] torch.FloatTensor of values in {0, 1}. Original image size.
                     - crop_indices: a [N, 4] torch.LongTensor. xmin, ymin, xmax, ymax.
            other_args: a Python dictionary with the following keys:
                          - rgb_img_features : output of gc.extract_rgb_img_features
                          - xyz_img : a [3, H, W] torch.FloatTensor

        Returns:
            a [E] torch.FloatTensor with values in [0,1]. E = # of edges in graph
        """

        # Don't consider the background node
        E = graph.edge_index.shape[1]
        non_bg_edge = torch.all(graph.edge_index != 0, dim=0)  # [E]
        non_bg_edge_indices = torch.where(non_bg_edge)[0]  # [E_no_bg]
        non_bg_edge_index = graph.edge_index[:, non_bg_edge]  # [2, E_no_bg]
        graph_no_bg = gc.remove_bg_node(graph)  # edge index is now: [2, E_no_bg]

        # Run on every pair of nodes here
        union_graph = gc.get_edge_graph(graph_no_bg, 
                                        other_args['rgb_img_features'], 
                                        other_args['xyz_img'],
                                       )

        split_score_logits, boundary_logits = self.model(union_graph)
        boundary_probs = torch.sigmoid(boundary_logits)  # [E_no_bg, h, w]

        # Merge score by looking at split boundaries
        merge_scores = torch.zeros((E,), device=constants.DEVICE)
        for k, (i, j) in enumerate(non_bg_edge_index.T):  # k ranges over [0, E_no_bg)

            resized_m0 = gc.crop_tensor_to_nchw(graph.orig_masks[i],
                                                *union_graph.crop_indices[k],
                                                self.config['img_size'],
                                                mode='nearest')[0, 0]  # [h, w]
            resized_m1 = gc.crop_tensor_to_nchw(graph.orig_masks[j],
                                                *union_graph.crop_indices[k],
                                                self.config['img_size'],
                                                mode='nearest')[0, 0]  # [h, w]

            boundary_overlap = util_.mask_boundary_overlap(resized_m0.cpu().numpy(),
                                                           resized_m1.cpu().numpy(), d=2)  # [h, w]
            boundary_overlap = torch.from_numpy(boundary_overlap).float().to(self.device)

            if torch.sum(boundary_overlap) == 0:
                merge_scores[non_bg_edge_indices[k]] = 0
            else:
                merge_scores[non_bg_edge_indices[k]] = (1 - torch.sum(boundary_probs[k] * boundary_overlap) /
                                                        torch.sum(boundary_overlap))

        return merge_scores

