import os
import gc as garbage_collection
import copy
from collections import OrderedDict
import json

import numpy as np
import cv2
from skimage.morphology import disk

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean

from . import constants
from . import graph_construction as gc
from . import merge_split_networks as msn
from .util import utilities as util_


class SampleTreeNode:
    """A simple tree implementation."""

    def __init__(self, graph, parent=None, _id='0'):
        self.graph = graph
        self.parent = parent
        self.children = []
        self.id = _id  # _id is used for visualization only

    def depth(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.depth()

    def max_depth(self):
        return max([x.depth() for x in self.leaves()])

    def add_child(self, tree_node):
        _id = self.id + f'_{len(self.children)}'
        tree_node = SampleTreeNode(tree_node, parent=self, _id=_id)
        self.children.append(tree_node)

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()

    def is_leaf(self):
        return len(self.children) == 0

    def leaves(self):
        if self.is_leaf():
            return [self]
        else:
            all_leaves = []
            for child in self.children:
                all_leaves += child.leaves()
            return all_leaves

    def all_nodes(self):
        all_nodes = [self]
        for child in self.children:
            all_nodes += child.all_nodes()
        return all_nodes

    def all_graphs(self):
        return [x.graph for x in self.all_nodes()]

    def num_graphs(self):
        return len(self.all_graphs())

    def num_total_nodes(self):
        num_nodes = sum([x.mask.shape[0] for x in self.all_graphs()])
        return num_nodes

    def num_total_edges(self):
        num_edges = sum([x.edge_index.shape[1] for x in self.all_graphs()])
        return num_edges

    def graph_in_tree(self, graph):
        return any([util_.graph_eq(graph, node.graph) for node in self.all_nodes()])

    def __repr__(self):
        return f"SampleTreeNode {self.id}: {repr(self.graph)}"


def sample_merges(graph_, rgb_img_features, xyz, 
                  merge_scores, num_merges, threshold,
                  gc_neighbor_dist, padding_config,
                  **kwargs):
    """Sample merges.

    Args:
        graph_: a torch_geometric.data.Batch instance with attributes:
                  - rgb: a [N x C_app] torch.FloatTensor of rgb features
                  - depth: a [N x 3 x H' x W'] torch.FloatTensor
                  - mask: a [N x 1 x H' x W'] torch.FloatTensor
                  - orig_masks: a [N x H x W] torch.FloatTensor of original masks
                  - crop_indices: a [N, 4] torch.LongTensor. xmin, ymin, xmax, ymax.
        rgb_img_features: an OrderedDict of image features. Output of gc.extract_rgb_img_features()
        xyz_img: a [3, H, W] torch.FloatTensor. 3D point cloud from camera frame of reference
        merge_scores: a [E] torch.FloatTensor with values in [0, 1]. Output of
                      MergeBySplitWrapper.merge_scores_via_splitnet().
        num_merges: Maximum number of merges allowed.
        threshold: Minimum merge score required to consider the merge.
        gc_neighbor_dist: Distance threshold for connecting nodes in new graph
        padding_config: a Python dictionary with padding parameters.

    Returns:
        boolean of whether merge operation was successful.
        a torch_geometric.data.Data instance.
    """

    # Sort scores, consider only the ones above a certain threshold
    sorted_scores, score_indices = torch.sort(merge_scores, descending=True)
    num_potential_merges = torch.sum(sorted_scores > threshold)
    if num_potential_merges == 0:  # Nothing to merge
        return False, None
    score_indices = score_indices[:num_potential_merges]

    # Select some masks to merge
    merge_indices = torch.zeros((0, 2), dtype=torch.long, device=constants.DEVICE)
    leftover_merge_scores = merge_scores[score_indices].clone()  # [num_potential_merges]
    leftover_merge_indices = graph_.edge_index.T[score_indices].clone()  # [num_potential_merges, 2]
    while merge_indices.shape[0] < num_merges and leftover_merge_indices.shape[0] > 0:

        # Sample mask indices to merge
        sample_idx = torch.multinomial(leftover_merge_scores, 1)
        to_merge = leftover_merge_indices[sample_idx]
        merge_indices = torch.cat([to_merge, merge_indices], dim=0)

        # Get edges that are not conflicting with this merge
        temp = torch.all(
            ((leftover_merge_indices != to_merge[0,0]) & 
             (leftover_merge_indices != to_merge[0,1])),
            dim=1) 
        leftover_merge_indices = leftover_merge_indices[temp]
        leftover_merge_scores = leftover_merge_scores[temp]

    # Merge the masks
    new_masks = graph_.orig_masks.clone()
    for i,j in merge_indices:
        new_masks[i] = torch.clamp(graph_.orig_masks[i] + graph_.orig_masks[j], max=1)
        new_masks[j] = 0

    # Create new graph
    new_masks = new_masks[1:]  # Get rid of BG mask
    new_masks = util_.convert_mask_NHW_to_HW(new_masks.float(), start_label=constants.OBJECTS_LABEL)
    new_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, new_masks, 
                                                neighbor_dist=gc_neighbor_dist,
                                                padding_config=padding_config)

    return True, new_graph


def sample_splits(graph_, rgb_img_features, xyz,
                  split_scores, num_splits, threshold,
                  gc_neighbor_dist, padding_config,
                  min_pixels_thresh=100,
                  **kwargs):
    """Sample Splits.

    Args:
        graph_: a torch_geometric.data.Batch instance with attributes:
                  - rgb: a [N x C_app] torch.FloatTensor of rgb features
                  - depth: a [N x 3 x H' x W'] torch.FloatTensor
                  - mask: a [N x 1 x H' x W'] torch.FloatTensor
                  - orig_masks: a [N x H x W] torch.FloatTensor of original masks
                  - crop_indices: a [N, 4] torch.LongTensor. xmin, ymin, xmax, ymax.
        rgb_img_features: an OrderedDict of image features. Output of gc.extract_rgb_img_features()
        xyz_img: a [3, H, W] torch.FloatTensor. 3D point cloud from camera frame of reference
        split_scores: a [N] torch.FloatTensor with values in [0, 1]. Output of
                      SplitNetWrapper.split_scores_via_sampling_splits().
        num_splits: Maximum number of splits allowed.
        threshold: Minimum split score required to consider the split.
        gc_neighbor_dist: Distance threshold for connecting nodes in new graph
        padding_config: a Python dictionary with padding parameters.
        min_pixels_thresh: int.

    Returns:
        boolean of whether merge operation was successful.
        a torch_geometric.data.Data instance.
    """
    H, W = graph_.orig_masks.shape[1:3]

    # Sort scores, consider only the ones above a certain threshold
    sorted_scores, score_indices = torch.sort(split_scores, descending=True)
    num_potential_splits = torch.sum(sorted_scores > threshold)
    if num_potential_splits == 0:  # Nothing to split
        return False, None
    score_indices = score_indices[:num_potential_splits]          

    split_inds = torch.zeros(graph_.orig_masks.shape[0]).bool()

    # Sample some masks to split
    new_masks = torch.zeros((0,H,W), device=constants.DEVICE) # Shape: [N x H x W]
    leftover_split_scores = split_scores[score_indices]
    leftover_split_indices = score_indices

    while torch.sum(split_inds) < num_splits and leftover_split_indices.shape[0] > 0:

        # Sample split index
        sample_idx = torch.multinomial(leftover_split_scores, 1)
        split_idx = leftover_split_indices[sample_idx][0]

        # Resize to original HxW size. See if it's valid (at least 2 masks, and all at least min_pixels_thresh)
        split_mask = msn.split_mask_upsample(graph_.orig_masks[split_idx],
                                             graph_.split[split_idx],  # This was stored when we ran SplitNet
                                             graph_.crop_indices[split_idx])
        split_mask = util_.convert_mask_HW_to_NHW(split_mask)
        split_mask = split_mask[split_mask.sum(dim=(1,2)) >= min_pixels_thresh]  # filter out masks that are too small
        if split_mask.shape[0] >= 2:
            new_masks = torch.cat([new_masks, split_mask], dim=0)
            split_inds[split_idx] = True

        # Get leftover potential splits
        temp = torch.ones(leftover_split_scores.shape[0]).bool()
        temp[sample_idx] = False
        leftover_split_indices = leftover_split_indices[temp]
        leftover_split_scores = leftover_split_scores[temp]

    if split_inds.sum() == 0:
        return False, None

    # Keep the old un-split masks, add the new ones
    new_masks = torch.cat([graph_.orig_masks[~split_inds], new_masks], dim=0)
    new_masks = util_.filter_out_empty_masks_NHW(new_masks)

    # Create new graph
    new_masks = new_masks[1:]  # Get rid of BG mask
    new_masks = util_.convert_mask_NHW_to_HW(new_masks.float(), start_label=constants.OBJECTS_LABEL)
    new_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, new_masks, 
                                                neighbor_dist=gc_neighbor_dist,
                                                padding_config=padding_config)

    return True, new_graph


def sample_deletes(graph_, rgb_img_features, xyz, 
                   delete_scores, num_deletes, threshold,
                   gc_neighbor_dist, padding_config,
                   **kwargs):
    """Sample Deletes.

    Args:
        graph_: a torch_geometric.data.Batch instance with attributes:
                  - rgb: a [N x C_app] torch.FloatTensor of rgb features
                  - depth: a [N x 3 x H' x W'] torch.FloatTensor
                  - mask: a [N x 1 x H' x W'] torch.FloatTensor
                  - orig_masks: a [N x H x W] torch.FloatTensor of original masks
                  - crop_indices: a [N, 4] torch.LongTensor. xmin, ymin, xmax, ymax.
        rgb_img_features: an OrderedDict of image features. Output of gc.extract_rgb_img_features()
        xyz_img: a [3, H, W] torch.FloatTensor. 3D point cloud from camera frame of reference
        delete_scores: a [N] torch.FloatTensor with values in [0, 1]. Output of
                       DeleteNetWrapper.delete_scores().
        num_deletes: Maximum number of deletes allowed.
        threshold: Minimum delete score required to consider the delete.
        gc_neighbor_dist: Distance threshold for connecting nodes in new graph
        padding_config: a Python dictionary with padding parameters.

    Returns:
        boolean of whether merge operation was successful.
        a torch_geometric.data.Data instance.
    """

    # Sort scores, consider only the ones above a certain threshold
    sorted_scores, score_indices = torch.sort(delete_scores, descending=True)
    num_potential_deletes = torch.sum(sorted_scores > threshold)
    if num_potential_deletes == 0 and torch.all(~graph_.added):  # Nothing to delete
        return False, None
    score_indices = score_indices[:num_potential_deletes]          

    delete_inds = torch.zeros(graph_.orig_masks.shape[0]).bool()

    # Sample some masks to delete
    leftover_delete_scores = delete_scores[score_indices]
    leftover_delete_indices = score_indices

    while torch.sum(delete_inds) < num_deletes and leftover_delete_indices.shape[0] > 0:

        # Sample delete index
        sample_idx = torch.multinomial(leftover_delete_scores, 1)
        delete_idx = leftover_delete_indices[sample_idx][0]
        delete_inds[delete_idx] = True

        # Get leftover potential deletes
        temp = torch.ones(leftover_delete_scores.shape[0]).bool()
        temp[sample_idx] = False
        leftover_delete_indices = leftover_delete_indices[temp]
        leftover_delete_scores = leftover_delete_scores[temp]

    # If the deleting only undoes the potential adds, consider the sampling to be a failure
    if torch.all(delete_inds == graph_.added):
        return False, None

    # Keep the un-deleted masks
    new_masks = graph_.orig_masks[~delete_inds]

    # Create new graph
    new_masks = new_masks[1:]  # Get rid of BG mask
    new_masks = util_.convert_mask_NHW_to_HW(new_masks.float(), start_label=constants.OBJECTS_LABEL)
    new_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, new_masks, 
                                                neighbor_dist=gc_neighbor_dist,
                                                padding_config=padding_config)

    return True, new_graph


def sample_tree_CEM(graph,
                    rgb_img_features,
                    xyz_img,
                    fg_mask,
                    sample_operator_networks,
                    sgs_net=None,
                    branch_factor=2,
                    num_iterations=3,
                    num_perturbations=3,
                    perturbation_threshold=0.5,
                    max_nodes_in_batch=150,
                    max_edges_in_batch=600,
                    verbose=False,
                    **kwargs,
                   ):
    """Build the sample tree of segmentation graphs via CEM.

    Args:
        graph: a torch_geometric.Data instance. The root of the tree.
        rgb_img_features: an OrderedDict of image features. Output of gc.extract_rgb_img_features()
        xyz_img: a [3, H, W] torch.FloatTensor. 3D point cloud from camera frame of reference
        fg_mask: a [H, W] torch.FloatTensor with values in {0, 1}. 
        sample_operator_networks: dict. Has the following keys:
            - mergenet_wrapper: MergeNetWrapper instance.
            - splitnet_wrapper: SplitNetWrapper instance.
            - deletenet_wrapper: DeleteNetWrapper instance.
        sgs_net: an instance of SGSNetWrapper or None.
            If None, just sample the graph. If not None, use it to expand the leaves (only add leaf
            if leaf is better than parent).
        branch_factor: int. Maximum number of children per node.
        num_iterations: int. Number of sample tree expansion iterations.
        num_perturbations: int. Number of perturbations per sample (e.g. 2 splits per sample).
        perturbation_threshold: float in [0, 1]. Used to threshold merge/split/delete scores.
        max_nodes_in_batch: int.
        max_edges_in_batch: int.
        verbose: bool.
        kwargs: dict. Keyword arguments. Meant to be passed to sample operation functions.
            Should contain:
            - gc_neighbor_dist
            - padding_config

    Returns:
        a SampleTreeNode instance. This is the root of the sample tree.
    """
    def clean_up_memory():
        import gc as garbage_collection
        garbage_collection.collect()
        torch.cuda.empty_cache()

    def print_tree_stats():
        if verbose:
            print(f"#graphs: {len(sample_tree.all_graphs())}, "
                  f"#nodes: {sample_tree.num_total_nodes()}, "
                  f"#edges: {sample_tree.num_total_edges()}")

    def _apply_sample_func(sample_func, *args, **kwargs):
        return sample_func(*args, **kwargs)

    sample_func_to_name = {
        sample_merges  : 'merge',
        sample_splits  : 'split',
        sample_deletes : 'delete',
    }
    for network in sample_operator_networks.values():
        network.eval_mode()

    # Instantiate sample tree with root node
    sample_tree = SampleTreeNode(graph)

    for _ in range(num_iterations):
        for leaf in sample_tree.leaves():
            graph_ = leaf.graph

            potential_sample_funcs = list(sample_func_to_name.keys())

            # SplitNet/MergeNet checks:
            N = graph_.orig_masks.shape[0]
            if N <= 1:  # Only BG mask is present
                if verbose:
                    print('Only background mask is present...')
                continue
            else:
                neighbor_edge_index = util_.neighboring_mask_indices(
                    gc.remove_bg_node(graph_).orig_masks, neighbor_dist=1).to(constants.DEVICE).T  # [2 x E]
                if neighbor_edge_index.shape[1] == 0:
                    if verbose:
                        print(f'had to remove merging, |E| = {neighbor_edge_index.shape[1]}')
                    potential_sample_funcs.remove(sample_merges)
            # Note: DeleteNet can always be run, since it considers adding masks

            # Call sample operation networks, save the scores, see if any can be used.
            scores_dict = dict()
            graph_sample_input_dict = dict()
            potential_sample_func_names = [sample_func_to_name[sample_func]
                                           for sample_func in potential_sample_funcs]
            
            if 'merge' in potential_sample_func_names:

                # Make a copy of the graph and edit the edge_index
                merge_input_graph = copy.deepcopy(graph_)
                merge_input_graph.edge_index = util_.neighboring_mask_indices(
                    graph_.orig_masks, neighbor_dist=1).to(constants.DEVICE).T  # Only consider neighboring masks
                with torch.no_grad():
                    merge_scores = (sample_operator_networks['mergenet_wrapper'].
                                    merge_scores_via_splitnet(merge_input_graph,
                                                              {'rgb_img_features' : rgb_img_features, 
                                                               'xyz_img' : xyz_img}
                                                             )
                                    )
                scores_dict['merge'] = merge_scores
                graph_sample_input_dict['merge'] = merge_input_graph
                if torch.sum(merge_scores > perturbation_threshold) == 0:
                    if verbose:
                        print("Removing MERGING due to low scores.")
                    potential_sample_funcs.remove(sample_merges)

            if 'split' in potential_sample_func_names:

                # Make a copy of the graph. Splits/paths will be written to this graph from SplitNet
                split_input_graph = copy.deepcopy(graph_)
                with torch.no_grad():
                    split_scores = (sample_operator_networks['splitnet_wrapper'].
                                    split_scores_via_sampling_splits(split_input_graph))
                scores_dict['split'] = split_scores
                graph_sample_input_dict['split'] = split_input_graph
                if torch.sum(split_scores > perturbation_threshold) == 0:
                    if verbose:
                        print("Removing SPLITTING due to low scores.")
                    potential_sample_funcs.remove(sample_splits)

            if 'delete' in potential_sample_func_names:

                # Construct graph with added nodes. Mark them as added
                orig_masks = graph_.orig_masks[1:]  # no BG mask
                new_masks = (sample_operator_networks['deletenet_wrapper'].
                             get_new_potential_masks(orig_masks, fg_mask))
                delete_input_graph = gc.construct_segmentation_graph(
                    rgb_img_features, xyz_img, new_masks, create_edge_indices=False)
                delete_input_graph.added = torch.ones(new_masks.shape[0]+1).bool()
                delete_input_graph.added[:orig_masks.shape[0] + 1] = False

                with torch.no_grad():
                    delete_scores = (sample_operator_networks['deletenet_wrapper'].
                                     delete_scores(delete_input_graph))
                scores_dict['delete'] = delete_scores
                graph_sample_input_dict['delete'] = delete_input_graph
                if (torch.sum(delete_scores > perturbation_threshold) == 0 and
                    torch.all(~delete_input_graph.added)):
                    if verbose:
                        print("Removing DELETING due to low scores.")
                    potential_sample_funcs.remove(sample_deletes)

            clean_up_memory()
            if len(potential_sample_funcs) == 0:  # can't merge/split/delete
                continue

            sample_funcs = np.random.choice(potential_sample_funcs, size=branch_factor)

            # Call sample functions in parallel
            args_list = []
            for j in range(branch_factor):

                sample_func = sample_funcs[j]
                graph_input = graph_sample_input_dict[sample_func_to_name[sample_func]]  # Potentially edited graph for input to sample_func
                scores = scores_dict[sample_func_to_name[sample_func]]

                args_list.append(
                    (sample_func, 
                     graph_input,
                     rgb_img_features,
                     xyz_img,
                     scores,
                     num_perturbations,
                     perturbation_threshold,
                    )
                )

            graph_samples = util_.parallel_map(_apply_sample_func, *zip(*args_list), **kwargs)

            if sgs_net:
                successful_indices = [ind for ind in range(len(graph_samples)) if graph_samples[ind][0]]
                successful_graph_samples = [graph_samples[ind][1] for ind in successful_indices]
                if successful_indices:
                    comparison_scores = sgs_net.compare_graphs([graph_], successful_graph_samples, 
                                                               {'rgb_img_features' : rgb_img_features, 
                                                                'xyz_img' : xyz_img,
                                                               }
                                                              )  # [1, len(successful_indices)]
                    comparison_scores = comparison_scores[0]  # [len(successful_indices)]. boolean array.
                    for enum_index, ind in enumerate(successful_indices):
                        graph_samples[ind] = (comparison_scores[enum_index], graph_samples[ind][1])                           
                        # Note: since we are only doing this for successful samples, this is essentially AND-ing
                        #       with the score from the sgs_net.
                    clean_up_memory()

            # Add success new graph samples to sample tree
            for j, (success, new_graph) in enumerate(graph_samples):

                if verbose:
                    print(f"Node {leaf.id}. "
                          f"Potential child {len(leaf.children)}: "
                          f"{sample_func_to_name[args_list[j][0]]}. Success: {success}")

                if success:
                    leaf.add_child(new_graph)
                    if (sample_tree.num_total_edges() >= max_edges_in_batch or 
                        sample_tree.num_total_nodes() >= max_nodes_in_batch):
                        leaf.children.pop(-1)
                        print_tree_stats()
                        return sample_tree

    print_tree_stats()
    return sample_tree




def graph_seg_bmap(graph):
    """Compute boundary map of graph segmentation masks.
    
    Args:
        masks: [N, H, W] np.ndarray. Values in {0, 1}.
    
    Returns:
        [H, W] np.ndarray. Values in {0, 1}.
    """

    H, W = graph.orig_masks.shape[1:3]
    boundary_map = np.zeros((H, W))

    masks = graph.orig_masks.cpu().numpy()
    for mask in masks[1:]:  # don't consider BG mask
        one_px_bmap = util_.seg2bmap(mask)
        dilated_bmap = cv2.dilate(one_px_bmap, disk(3), iterations=1)
        boundary_map = np.logical_or(boundary_map, dilated_bmap)
        
    return boundary_map


class SampleTreeCEMWrapper:

    def __init__(self,
                 rn50_fpn,
                 sample_operator_networks,
                 sgs_net_wrapper,
                 sample_tree_params,
                 ):

        """Initialization for SampleTreeRefiner, which uses SGSNet.

        Args:
            rn50_fpn: A pretrained ResNet50+FPN model. Instance of BackboneWithFPN.
            sample_operator_networks: dict. Has the following keys:
                - mergenet_wrapper: MergeNetWrapper instance.
                - splitnet_wrapper: SplitNetWrapper instance.
                - deletenet_wrapper: DeleteNetWrapper instance.
            sgs_net_wrapper: instance of gn.SGSNetWrapper.
            sample_tree_params: dict with the following keys:
                - branch_factor
                - num_iterations
                - num_perturbations
                - perturbation_threshold
                - max_nodes_in_batch
                - max_edges_in_batch
        """
        self.rn50_fpn = rn50_fpn
        self.sample_operator_networks = sample_operator_networks
        self.sgs_net_wrapper = sgs_net_wrapper
        self.sample_tree_params = sample_tree_params

    def clean_up_memory(self):
        # Manually clean GPU memory
        garbage_collection.collect()
        torch.cuda.empty_cache()

    def run_on_batch(self, batch, verbose=False):
        """Sample perturbations to the graph and score them.

        Args:
            batch: dict with the following keys:
                - rgb : a [1, 3, H, W] torch.FloatTensor
                - xyz : a [1, 3, H, W] torch.FloatTensor
                - seg_masks : a [1, H, W] torch tensor with values in {0, 1, 2, ...}
                - fg_mask: a [1, H, W] torch tensor with values in {0, 1}
        """
        self.sgs_net_wrapper.eval_mode()
        self.sgs_net_wrapper.send_batch_to_device(batch)

        rgb = batch['rgb'][0]  # [3, H, W]
        xyz = batch['xyz'][0]  # [3, H, W]
        seg_masks = batch['seg_masks'][0]  # [H, W]
        fg_mask = batch['fg_mask'][0]  # [H, W]

        with torch.no_grad():

            # Create base graph
            rgb_img_features = gc.extract_rgb_img_features(self.rn50_fpn, rgb)
            base_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, seg_masks)
            self.clean_up_memory()

            # Sample the tree
            kwargs = self.sample_tree_params.copy()
            kwargs['gc_neighbor_dist'] = self.sgs_net_wrapper.config['gc_neighbor_dist']
            kwargs['padding_config'] = None
            sample_tree = sample_tree_CEM(base_graph,
                                          rgb_img_features,
                                          xyz,
                                          fg_mask,
                                          self.sample_operator_networks,
                                          self.sgs_net_wrapper,
                                          verbose=verbose,
                                          **kwargs,
                                         )
            self.clean_up_memory()

        # Score any graphs that are not already scored. This can happen in edge cases, e.g. a root that never expands.
        unscored_graphs = [g for g in sample_tree.all_graphs() if 'sgs_net_score' not in g.keys]
        if unscored_graphs:
            other_args = {'rgb_img_features' : rgb_img_features, 'xyz_img' : xyz}
            scores = self.sgs_net_wrapper.run_on_graphs(unscored_graphs, other_args).cpu().numpy()
            for i in range(len(unscored_graphs)):
                unscored_graphs[i].sgs_net_score = scores[i]

        return sample_tree


    def best_graph(self, graph_list):
        scores = np.array([g.sgs_net_score for g in graph_list])
        best_graph_index = np.argmax(scores)
        best_graph = graph_list[best_graph_index]
        best_score = scores[best_graph_index]
        return best_graph, best_score

    def contour_uncertainties(self, sample_tree, scores=None):
        """Compute contour uncertainties.

        Args:
            sample_tree: SampleTreeNode instance with N nodes
            scores: a [N] torch.FloatTensor with values in [0, 1]. Scores of sample_tree
                w.r.t. DFS ordering (same ordering as sample_tree.all_graphs())

        Returns:
            a [H, W] torch.FloatTensor with values in [0, 1].
            a [H, W] torch.FloatTensor
        """
        leaf_nodes = sample_tree.leaves()
        leaf_boundary_maps = [graph_seg_bmap(node.graph) for node in leaf_nodes]
        leaf_boundary_maps = np.stack(leaf_boundary_maps)  # [L, H, W], where L = num_leafs

        if scores is not None:

            leaf_indices = [node.is_leaf() for node in sample_tree.all_nodes()]
            leaf_scores = scores[leaf_indices, None, None]  # [L, 1, 1]
            leaf_scores = leaf_scores.cpu().numpy()
            contour_mean = (leaf_boundary_maps * leaf_scores).sum(axis=0) / leaf_scores.sum()  # [H, W]
            contour_std = (((leaf_boundary_maps - contour_mean[None])**2 * leaf_scores).sum(axis=0) / 
                            leaf_scores.sum())

        else:

            contour_mean = leaf_boundary_maps.mean(axis=0)
            contour_std = leaf_boundary_maps.std(axis=0)

        return contour_mean, contour_std

    def evaluate_once(self, dl, save_dir):
        """Evaluate the model on a dataset, save the results to disk.

        Args:
            dl: An instance of Dataset class defined in data_loader.py.
            save_dir: string.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        from tqdm import tqdm
        progress = tqdm(dl)
        for batch in progress:

            # Check if this was already computed or not. Assumes batch size is 1.
            path = batch['label_abs_path'][0]
            base_abs_path, abs_path_fname = path.rsplit('/', 1)
            file_path = os.path.join(save_dir, base_abs_path)
            file_name = os.path.join(file_path, abs_path_fname.replace('.pcd', '.png'))
            if os.path.exists(file_name):
                continue

            # Run SampleTreeCEM
            sample_tree = self.run_on_batch(batch, verbose=False)
            best_graph, _ = self.best_graph(sample_tree.all_graphs())
            best_seg = gc.remove_bg_node(best_graph).orig_masks
            best_seg = util_.convert_mask_NHW_to_HW(best_seg, start_label=constants.OBJECTS_LABEL).cpu().numpy()

            # Write results to disk
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            cv2.imwrite(file_name, best_seg.astype(np.uint8))

    def evaluate(self, dl, base_save_dir, num_evals):
        """Evaluate the model on a dataset multiple times, save the results to disk.
        """
        if not os.path.exists(base_save_dir):
            os.makedirs(base_save_dir)

        for i in range(num_evals):
            print(f"Eval {i+1}...")
            self.evaluate_once(dl, os.path.join(base_save_dir, f'{i:03d}'))

