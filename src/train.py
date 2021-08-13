import sys, os
import copy
from time import time
import yaml
import itertools
import gc as garbage_collection
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import torch_geometric
from torch.utils.tensorboard import SummaryWriter

from . import network_config as nc
from . import graph_construction as gc
from . import sample_tree_cem as stc
from . import data_augmentation as data_aug
from . import losses
from . import merge_split_networks as msn
from . import constants
from .util import utilities as util_


def randomly_merge_masks(graph,
                         rgb_img_features,
                         xyz,
                         num_merges,
                         padding_config,
                         gt_labels,
                         return_merge_indices=False,
                         neighbor_dist=10,
                         device=None):
    """Training procedure to randomly merge masks.

    Args:
        graph: a torch_geometric.Data instance.
        rgb_img_features: Dict of torch.Tensors. Output of gc.extract_rgb_img_features().
        xyz: a [3, H, W] torch.Tensor.
        num_merges: int. Number of merges to sample.
        padding_config: Dictionary with configuration for padding. Used for gc.construct_segmentation_graph().
        gt_labels: a [N_gt, H, W] torch.Tensor with values in {0, 1}.
        return_merge_indices: bool.
        neighbor_dist: int.
        device: None or str.

    Returns:
        a boolean whether the graph sampling was successful.
        Tuple: a torch_geometric.Data instance constructed from newly-sampled masks.
               a dict that maps from {0, ..., N'-1} (index of new_masks) to tuple of merge indices (for original masks).
    """
    if device is None:
        device = constants.DEVICE
    no_bg_masks = graph.orig_masks[1:]

    # Find corresponding GT label for each mask, compute which masks can be merged for a GT split
    mask_labels = util_.mask_corresponding_gt(no_bg_masks, gt_labels)
    indices = util_.neighboring_mask_indices(no_bg_masks, neighbor_dist=1, reduction_factor=1)
    indices = indices[mask_labels[indices[:,0]] != mask_labels[indices[:,1]]] # indices that can be merged

    # Select some labels to merge
    merge_indices = torch.zeros((num_merges, 2), dtype=torch.long, device=device)
    leftover_indices = indices.clone()
    i = 0
    while i < num_merges and leftover_indices.shape[0] > 0:

        to_merge = leftover_indices[torch.randint(leftover_indices.shape[0],[1])][0] # Sample mask indices to merge
        temp = torch.all((leftover_indices != to_merge[0]) & (leftover_indices != to_merge[1]), dim=1) # Compute neighboring indices that are not 
        leftover_indices = leftover_indices[temp]

        # If these two masks have been split, don't merge them
        if ('status' in graph.keys and 
            graph.status[to_merge[0]+1] == constants.NODE_STATUS_SPLIT and  # +1 to account for BG node
            graph.status[to_merge[1]+1] == constants.NODE_STATUS_SPLIT):
            continue

        merge_indices[i] = to_merge # Store it
        i += 1

    merge_indices = merge_indices[:i]
    num_merges = i

    if num_merges == 0:
        if return_merge_indices:
            return False, graph, {}  # If failure, just return the original graph
        else:
            return False, graph

    # Merge the masks into the i^th mask
    new_masks = no_bg_masks.clone().bool()
    for i,j in merge_indices:
        new_masks[i] = new_masks[i] | new_masks[j]
        new_masks[j] = 0
    new_masks = new_masks.float()

    # Build the graph
    filtered_new_masks = util_.filter_out_empty_masks_NHW(new_masks)
    new_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, filtered_new_masks,
                                                neighbor_dist=neighbor_dist,
                                                padding_config=padding_config,
                                                device=device)

    if return_merge_indices:

        # Compute mapping from index of new_masks to tuple of merge indices (of no_bg_masks).
        mapping = {}
        keep_inds = new_masks.sum(dim=(1,2)) > 0.5
        for i,j in merge_indices:
            new_index = keep_inds[:i].sum()  # basically a cumsum
            mapping[new_index] = (i, j)

        return True, new_graph, mapping

    else:
        return True, new_graph


def randomly_split_masks(graph,
                         rgb_img_features,
                         xyz,
                         num_splits,
                         padding_config,
                         neighbor_dist=10,
                         device=None):
    """Training procedure to randomly split masks with straight lines.

    Splitting a node is done by sampling a random line in [-1,1]^2.
        Line is represented by w'x + b = 0.

    Args:
        graph: a torch_geometric.Data instance.
        rgb_img_features: Dict of torch.Tensors. Output of gc.extract_rgb_img_features().
        xyz: a [3, H, W] torch.Tensor.
        num_splits: int. Number of splits to sample.
        padding_config: Dictionary with configuration for padding. Used for gc.construct_segmentation_graph().
        neighbor_dist: int.
        device: None or str.

    Returns:
        a boolean whether the graph sampling was successful.
        a torch_geometric.Data instance constructed from newly-sampled masks.
    """
    if device is None:
        device = constants.DEVICE

    def sample_line():

        # Sample w by sampling an angle theta
        theta = torch.rand(1) * 2 * np.pi
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                [torch.sin(theta), torch.cos(theta)]],
                               dtype=torch.float,
                              )
        x_axis = torch.tensor([1,0],
                              dtype=torch.float,
                             )
        w = torch.mv(rot_mat, x_axis)

        return w

    def split_masks(ind):
        w = sample_line()
        H, W = no_bg_masks.shape[1:]

        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(no_bg_masks[ind])
        x_offset = tdist.Uniform(x_min + .1 * (x_max-x_min+1), x_min + .9 * (x_max-x_min+1)).sample()
        y_offset = tdist.Uniform(y_min + .1 * (y_max-y_min+1), y_min + .9 * (y_max-y_min+1)).sample()
        offset = torch.tensor([y_offset, x_offset])

        m_indices = util_.torch_moi(H,W) - offset[:,None,None]
        m_indices = m_indices.permute(1,2,0) # Shape: [H x W x 2]
        m_indices[..., 1] *= -1 # So x-axis points right, y-axis points up
        line_mask = (torch.mv(m_indices.reshape(-1,2), w)).reshape(H,W) >= 0
        line_mask = line_mask.to(device) # Shape: [H x W]

        new_mask1 = (no_bg_masks[ind].bool() & line_mask).float() # Shape: [H x W]
        new_mask2 = (no_bg_masks[ind].bool() & ~line_mask).float() # Shape: [H x W]

        return new_mask1, new_mask2


    no_bg_masks = graph.orig_masks[1:]
    N, H, W = no_bg_masks.shape
    new_masks = torch.zeros((0,H,W), device=device) # Shape: [N x H x W]

    split_inds = torch.randperm(N)[:num_splits]
    keep_inds = torch.ones(N).bool()

    # Sample lines to split the masks
    for split_ind in split_inds:

        valid_sizes = False
        num_tries = 0
        while not valid_sizes:
            if num_tries >= 20:
                break
            new_mask1, new_mask2 = split_masks(split_ind)
            valid_sizes = 1/8 < new_mask1.sum()/new_mask2.sum() < 8 # ratio must be within a range
            num_tries += 1

        if valid_sizes:
            new_masks = torch.cat([new_masks, new_mask1.unsqueeze(0), new_mask2.unsqueeze(0)], dim=0)
            keep_inds[split_ind] = False

    if keep_inds.sum() == N:
        return False, graph  # If failure, just return the original graph

    new_masks = torch.cat([no_bg_masks[keep_inds], new_masks], dim=0)

    # Build graph and edit status
    new_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, new_masks, 
                                                neighbor_dist=neighbor_dist,
                                                padding_config=padding_config,
                                                device=device)
    # Keep track of whether nodes have been split or added
    new_graph.status = torch.zeros(new_graph.mask.shape[0], dtype=torch.long,
                                   device=device) 
    if 'status' in graph.keys:
        new_graph.status[1:keep_inds.sum()+1] = graph.status[1:][keep_inds]  # +1 to account for BG node
    new_graph.status[keep_inds.sum()+1:] = constants.NODE_STATUS_SPLIT  # +1 to account for BG node

    return True, new_graph


def randomly_add_masks(graph,
                       rgb_img_features,
                       xyz,
                       num_adds,
                       padding_config,
                       config,
                       neighbor_dist=10,
                       device=None):
    """Training procedure to randomly add masks.

    Args:
        graph: a torch_geometric.Data instance.
        rgb_img_features: Dict of torch.Tensors. Output of gc.extract_rgb_img_features().
        xyz: a [3, H, W] torch.Tensor.
        num_adds: int. Number of adds to sample.
        padding_config: Dictionary with configuration for padding. Used for gc.construct_segmentation_graph().
        config: dict. See training YAML for more info.
        neighbor_dist: int.
        device: None or str.

    Returns:
        a boolean whether the graph sampling was successful.
        a torch_geometric.Data instance constructed from newly-sampled masks.
    """
    if device is None:
        device = constants.DEVICE

    no_bg_masks = graph.orig_masks[1:]
    N, H, W = no_bg_masks.shape

    new_masks = no_bg_masks.clone()
    perturb_indices = torch.randperm(N)[:num_adds]

    for idx in perturb_indices:

        morphed_label = new_masks[idx].cpu().numpy()

        # Sample rotation
        if torch.rand(1) < config['rate_of_rotation']:
            morphed_label = data_aug.random_rotation(morphed_label, config)

        # Sample add/cut
        sample = torch.rand(1)
        if sample < config['rate_of_label_adding']:
            morphed_label = data_aug.random_add(morphed_label, config)
        elif sample < config['rate_of_label_adding'] + config['rate_of_label_cutting']:
            morphed_label = data_aug.random_cut(morphed_label, config)
 
        # Find a random spot to put the mask
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(morphed_label)
        x_sidelength = x_max - x_min + 1
        y_sidelength = y_max - y_min + 1
        x_start = torch.randint(0, W - x_sidelength + 1, size=[1])[0]
        y_start = torch.randint(0, H - y_sidelength + 1, size=[1])[0]
        temp = np.zeros_like(morphed_label)
        temp[y_start:y_start+y_sidelength, x_start:x_start+x_sidelength] = morphed_label[y_min:y_max+1, x_min:x_max+1]
        morphed_label = temp

        # Make sure this doesn't intersect with existing masks
        morphed_label[new_masks.sum(dim=0).cpu().numpy() > 0.5] = 0  # Erase everything in this mask so there's no pixel assigned to multiple masks

        # Make sure it's 1 component
        morphed_label = util_.largest_connected_component(morphed_label, connectivity=8)

        # Make sure it's big enough
        if morphed_label.sum() > config['min_pixels_thresh']:
            morphed_label = torch.from_numpy(morphed_label).to(device)
            new_masks = torch.cat([new_masks, morphed_label[None]], dim=0)

    if new_masks.shape[0] == N:
        return False, graph  # If failure, just return the original graph

    # Build graph and edit status
    new_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, new_masks, 
                                                neighbor_dist=neighbor_dist,
                                                padding_config=padding_config,
                                                device=device)
    # Keep track of whether nodes have been split or added
    new_graph.status = torch.zeros(new_graph.mask.shape[0], dtype=torch.long,
                                   device=device) 
    if 'status' in graph.keys:
        new_graph.status[1:N+1] = graph.status[1:]  # +1 to account for BG node
    new_graph.status[N+1:] = constants.NODE_STATUS_ADDED  # +1 to account for BG node

    return True, new_graph


def randomly_delete_masks(graph,
                          rgb_img_features,
                          xyz,
                          num_deletes,
                          padding_config,
                          neighbor_dist=10,
                          device=None):
    """Training procedure to randomly delete masks.

    Args:
        graph: a torch_geometric.Data instance.
        rgb_img_features: Dict of torch.Tensors. Output of gc.extract_rgb_img_features().
        xyz: a [3, H, W] torch.Tensor.
        num_deletes: int. Number of deletes to sample.
        padding_config: Dictionary with configuration for padding. Used for gc.construct_segmentation_graph().
        neighbor_dist: int.
        device: None or str.

    Returns:
        a [N', H, W] torch.Tensor with values in {0, 1}. N' is the new total number of masks.
    """
    if device is None:
        device = constants.DEVICE

    no_bg_masks = graph.orig_masks[1:]
    N = no_bg_masks.shape[0]
    selection_mask = torch.ones(N, device=device).bool()
    for idx in torch.randperm(N):
        if (~selection_mask).sum() >= num_deletes:
            break
        if ('status' in graph and
            graph.status[idx+1] == constants.NODE_STATUS_ADDED):  # +1 to account for BG node
            continue
        selection_mask[idx] = False

    if selection_mask.sum() == N or selection_mask.sum() == 0:
        return False, graph  # If failure, just return the original graph

    new_masks = no_bg_masks[selection_mask]
    new_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, new_masks, 
                                                neighbor_dist=neighbor_dist,
                                                padding_config=padding_config,
                                                device=device)

    return True, new_graph


class Trainer(ABC):

    def __init__(self, model_wrapper, rn50_fpn, config, misc={}):
        self.model_wrapper = model_wrapper
        self.rn50_fpn = rn50_fpn
        self.config = config
        self.misc = misc  # random things that a trainer may need 
        self.device = self.model_wrapper.device

        # Initialize stuff
        self.epoch_num = 0
        self.iter_num = 0
        self.infos = dict()

        # Other setup
        self.setup()

        # Initialize optimizer
        model_params = [p for p in self.model_wrapper.model.parameters() if p.requires_grad]
        if ('trainable_layer_names' in self.config and
            len(self.config['trainable_layer_names']) > 0):
            model_params = model_params + \
                           [p for p in self.rn50_fpn.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(model_params, lr=self.config['lr'])

        if self.config['load']:
            self.load(self.config['opt_filename'], 
                      self.config['model_filename'],
                      self.config['rn50_fpn_filename'],
                     )

        # Tensorboard stuff
        self.tb_writer = SummaryWriter(self.config['tb_directory'],
                                       flush_secs=self.config['flush_secs'])

        # Save config files
        model_config_filename = os.path.join(self.config['tb_directory'],
                                             self.model_wrapper.__class__.__name__ + '_config')
        with open(model_config_filename, 'w') as file:
            yaml.dump(nc.dictify(self.model_wrapper.config), file)
        train_config_filename = os.path.join(self.config['tb_directory'],
                                             self.__class__.__name__ + '_config')
        with open(train_config_filename, 'w') as file:
            yaml.dump(nc.dictify(self.config), file)


    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def save(self, name=None, save_dir=None):
        """ Save optimizer state, epoch/iter nums, loss information

            Also save model state
        """

        # Save optimizer stuff
        checkpoint = {
            'iter_num' : self.iter_num,
            'epoch_num' : self.epoch_num,
            'infos' : self.infos,
        }

        checkpoint['optimizer'] = self.optimizer.state_dict()

        if save_dir is None:
            save_dir = self.config['tb_directory']

        if name is None:
            filename = os.path.join(save_dir,
                                    self.__class__.__name__ + '_' \
                                    + self.model_wrapper.__class__.__name__ \
                                    + '_iter' + str(self.iter_num) \
                                    + '_checkpoint.pth')
        else:
            filename = os.path.join(save_dir, name + '_checkpoint.pth')
        torch.save(checkpoint, filename)


        # Save model stuff
        filename = os.path.join(save_dir,
                                (self.model_wrapper.__class__.__name__ +
                                 '_iter' + str(self.iter_num) +
                                 '_checkpoint.pth')
                                )
        self.model_wrapper.save(filename)

        # Save ResNet50 FPN
        if len(self.config['trainable_layer_names']) > 0:
            checkpoint = {'model' : self.rn50_fpn.state_dict()}
            filename = os.path.join(save_dir,
                                    (self.rn50_fpn.__class__.__name__ +
                                     '_iter' + str(self.iter_num) +
                                     '_checkpoint.pth')
                                    )
            torch.save(checkpoint, filename)

    def load(self, 
             opt_filename=None,
             model_filename=None,
             rn50_fpn_filename=None):
        """ Load optimizer state, epoch/iter nums, loss information

            Also load model state
        """

        # Load optimizer stuff
        if opt_filename is not None:
            checkpoint = torch.load(opt_filename)

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded optimizer")

            self.iter_num = checkpoint['iter_num']
            self.epoch_num = checkpoint['epoch_num']
            self.infos = checkpoint['infos']


        # Load model stuff
        if model_filename is not None:
            self.model_wrapper.load(model_filename)

        # Load ResNet50+FPN file (if training first layer)
        if rn50_fpn_filename is not None and \
           len(self.config['trainable_layer_names']) > 0:
            checkpoint = torch.load(rn50_fpn_filename)
            self.rn50_fpn.load_state_dict(checkpoint['model'])
            print("Loaded ResNet50+FPN")


class SGSNetTrainer(Trainer):

    def setup(self):

        self.load_rn50_fpn_and_node_encoder()

        # Losses
        self.losses = {
            'graph_loss' : nn.BCEWithLogitsLoss(),
        }

    def load_rn50_fpn_and_node_encoder(self):

        # Load pre-trained ResNet50+FPN, if it exists (it exists if it was modality-tuned)
        checkpoint = torch.load(self.config['rn50_fpn_filename'])
        self.rn50_fpn.load_state_dict(checkpoint['model'])
        print(f"Loaded ResNet50 + FPN from: {self.config['rn50_fpn_filename']}")

        # Load fixed NodeEncoder (from SplitNet)
        checkpoint = torch.load(self.config['splitnet_filename'])['model']
        state_dict = self.model_wrapper.model.state_dict()
        for name, parameter in self.model_wrapper.model.named_parameters():
            name_no_module = name.replace('module.', '')
            if 'node_encoder.encoders' in name and name_no_module in checkpoint:
                state_dict[name] = checkpoint[name_no_module]
                parameter.requires_grad_(False)
        self.model_wrapper.model.load_state_dict(state_dict)
        print(f"Loaded NodeEncoder from: {self.config['splitnet_filename']}")

    def clean_up_memory(self):
        garbage_collection.collect()
        torch.cuda.empty_cache()

    def construct_training_example(self, graph, rgb_img_features, xyz, labels, debug=False):
        """Training example construction for SGS-Net training.
        
        Args:
            graph: a Data instance with keys:
                     - rgb          [N, d_n]
                     - depth        [N, 3, H', W']
                     - mask         [N, 1, H',]
                     - orig_masks   [N, H, W]
                     - crop_indices [N, 4]
                     Note: graph is NOT a Batch(Data) instance. It's only 1 graph.
            rgb_img_features: an OrderedDict of ResNet50+FPN features. output of gc.extract_rgb_img_features
            xyz: a [3, H, W] torch.FloatTensor. Organized point cloud by backprojecting a depth map
            labels: a [H, W] torch.FloatTensor of values in {0,1}. N_gt = #GT objects

        Returns:
            a torch_geometric.data.Batch instance.
        """
        def _apply_sample_func(sample_func, args_list, **kwargs):
            return sample_func(*args_list, **kwargs)

        def sample_graph_tree_from_root(graph):
            """Sample the graph tree.

            Args:
                graph: a torch_geometric.Data instance. The root of the tree.

            Returns:
                a sample tree represented as a Python Dictionary. Each key is the depth, and 
                    each value is a list of graphs at that depth. Note that the parent pointers
                    are not present in this representation.
            """
            sample_func_to_name = {
                randomly_merge_masks  : 'merge',
                randomly_split_masks  : 'split',
                randomly_add_masks : 'add',
                randomly_delete_masks : 'delete',
            }

            # Instantiate sample tree with root node
            sample_tree = stc.SampleTreeNode(graph)

            for _ in range(self.config['num_tree_building_iterations']):
                for leaf in sample_tree.leaves():
                    graph_ = leaf.graph

                    potential_sample_funcs = list(sample_func_to_name.keys())

                    # Checks:
                    N = graph_.orig_masks.shape[0]
                    if N <= 1:  # Only BG mask is present
                        print('Only background mask is present...')
                        continue
                    else:
                        neighbor_edge_index = util_.neighboring_mask_indices(
                            gc.remove_bg_node(graph_).orig_masks, neighbor_dist=1).to(constants.DEVICE).T  # [2 x E]
                        if neighbor_edge_index.shape[1] < 1:
                            print(f'had to remove merging, |E| = {neighbor_edge_index.shape[1]}')
                            potential_sample_funcs.remove(randomly_merge_masks)

                    # Sample the perturbation functions
                    sample_funcs = np.random.choice(potential_sample_funcs, size=self.config['branch_factor'])

                    # Call sample functions in parallel
                    args_lists = []
                    for j in range(self.config['branch_factor']):

                        sample_func_name = sample_func_to_name[sample_funcs[j]]
                        num_perturbations = self.config['num_perturbations']  # Number of merges/splits/deletes
                        masks_input = graph_.orig_masks[1:]  # don't use BG masks

                        args_list = [graph_, rgb_img_features, xyz, num_perturbations, self.config['padding_config']]
                        if sample_func_name == 'merge':
                            args_list.append(labels_nhw)
                        elif sample_func_name == 'add':
                            args_list.append(self.config['random_add_config'])
                        args_lists.append(args_list)
                 
                    graph_samples = util_.parallel_map(_apply_sample_func, sample_funcs, args_lists,
                                                       **{'neighbor_dist' : self.model_wrapper.config['gc_neighbor_dist']})

                    # Add success new graph samples to sample tree
                    for j, (success, new_graph) in enumerate(graph_samples):
                        if success and not sample_tree.graph_in_tree(new_graph):
                            leaf.add_child(new_graph)
                            if (sample_tree.num_total_nodes() > max_nodes_in_sample_tree or 
                                sample_tree.num_total_edges() > max_edges_in_sample_tree or
                                sample_tree.num_graphs() > max_graphs_in_sample_tree):
                                leaf.children.pop(-1)
                                return sample_tree

            return sample_tree


        labels_nhw = util_.convert_mask_HW_to_NHW(labels, to_ignore=[])

        # Add GT graph in the list
        gt_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, labels, 
                                                   neighbor_dist=self.model_wrapper.config['gc_neighbor_dist'],
                                                   padding_config=self.config['padding_config'])
        max_nodes_in_sample_tree = self.config['max_nodes_in_batch']
        max_edges_in_sample_tree = self.config['max_edges_in_batch']
        max_graphs_in_sample_tree = self.config['max_graphs_in_batch']

        max_nodes_in_sample_tree -= gt_graph.mask.shape[0]
        max_edges_in_sample_tree -= gt_graph.edge_index.shape[1]
        max_graphs_in_sample_tree -=  1

        # Sample the graph tree
        sample_tree = sample_graph_tree_from_root(graph)
        sample_list = sample_tree.all_graphs()  # list of graphs
        sample_list += [gt_graph]

        # Remove status key
        for new_graph in sample_list:
            if 'status' in new_graph:
                del new_graph.status

        # Score each graph
        scores = util_.parallel_map(losses.compute_graph_score, 
                                    [g.orig_masks[1:] for g in sample_list],  # 1: to account for BG node
                                    gt_mask=labels)
        for i, new_graph in enumerate(sample_list):
            new_graph.y = torch.tensor([scores[i]], dtype=torch.float)

        batch_sample_list = gc.convert_list_to_batch(sample_list).to(constants.DEVICE)
        if debug:
            return batch_sample_list, sample_tree
        else:
            return batch_sample_list


    def train(self, num_epochs, data_loader):

        # Some stuff to keep track of
        batch_time = util_.AverageMeter()
        data_time = util_.AverageMeter()
        batch_sizes = util_.AverageMeter()
        total_losses = util_.AverageMeter()
        end = time()

        # Training mode
        self.model_wrapper.train_mode()

        for epoch_iter in range(num_epochs):
            for batch in data_loader:

                if self.iter_num >= self.config['max_iters']:
                    print("Reached maximum number of iterations...")
                    self.save()
                    return

                # Send everything to GPU
                self.model_wrapper.send_batch_to_device(batch)

                # measure data loading time
                data_time.update(time() - end)

                rgb = batch['rgb'][0]  # [3, H, W]
                xyz = batch['xyz'][0]  # [3, H, W]
                seg_masks = batch['seg_masks'][0]  # [H, W]
                labels = batch['foreground_labels'][0]  # [H, W]
                H, W = labels.shape

                # Check that base graph has object IDs
                obj_ids = torch.unique(seg_masks)
                obj_ids = obj_ids[obj_ids >= constants.OBJECTS_LABEL]
                # print(f"Number of nodes for base graph: {obj_ids.shape[0]}")
                if obj_ids.shape[0] == 0:
                    print("0 graph nodes... discarding this example")
                    continue

                self.clean_up_memory()

                # Construct base segmentation graph, GT segmentation graph
                rgb_img_features = gc.extract_rgb_img_features(self.rn50_fpn, rgb)
                base_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, seg_masks, 
                                                             neighbor_dist=self.model_wrapper.config['gc_neighbor_dist'],
                                                             padding_config=self.config['padding_config'])

                # Construct training example
                batch_graph = self.construct_training_example(base_graph, rgb_img_features, xyz, labels)
                B = (batch_graph.batch.max() + 1).item()

                # import ipdb; ipdb.set_trace()

                # Apply the model/loss
                other_args_input = {
                    'rgb_img_features' : rgb_img_features, 
                    'xyz_img' : xyz,
                    'padding_config' : self.config['padding_config'],
                }
                out = self.model_wrapper.model(batch_graph, other_args=other_args_input)  # [B]
                loss = self.losses['graph_loss'](out, batch_graph.y)

                ### Gradient descent ###
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                total_losses.update(loss.item(), B)
                batch_sizes.update(B, 1)

                # Record some information about this iteration
                batch_time.update(time() - end)
                end = time()

                # Record information every x iterations
                if self.iter_num % self.config['iter_collect'] == 0:
                    info = {'iter_num': self.iter_num,
                            'Batch Time': round(batch_time.avg, 3),
                            'Data Time': round(data_time.avg, 3),
                            'Batch Size': round(batch_sizes.avg, 3),
                            'loss': round(total_losses.avg, 7),
                            }
                    self.infos[self.iter_num] = info

                    # Tensorboard stuff
                    self.tb_writer.add_scalar('Loss/Total', info['loss'], self.iter_num)
                    self.tb_writer.add_scalar('Batch Size', info['Batch Size'], self.iter_num)
                    self.tb_writer.add_scalar('Time/per iter', info['Batch Time'], self.iter_num)
                    self.tb_writer.add_scalar('Time/data fetch', info['Data Time'], self.iter_num)

                    # Reset meters
                    batch_time = util_.AverageMeter()
                    data_time = util_.AverageMeter()
                    batch_sizes = util_.AverageMeter()
                    total_losses = util_.AverageMeter()
                    end = time()

                self.iter_num += 1

            self.epoch_num += 1
            self.save()

        self.save()


class SplitNetTrainer(Trainer):

    def setup(self):

        # Losses
        self.losses = {
            'split_score_loss' : losses.BCEWithLogitsLossWeighted(weighted=False),
            'boundary_loss' : losses.BCEWithLogitsLossWeighted(weighted=False),
        }

    def construct_training_example(self, graph, rgb_img_features, xyz, gt_labels):
        """Training example construction.

        Args:
            graph: a torch_geometric.Data instance.
            rgb_img_features: an OrderedDict of ResNet50+FPN features. output of gc.extract_rgb_img_features
            xyz: a [3, H, W] torch.FloatTensor. Organized point cloud by backprojecting a depth map
            gt_labels: a [H, W] torch.FloatTensor with values in {0, ..., N_gt}. N_gt = #GT objects

        Returns:
            a torch_geometric.Data instance
            a [K] torch.FloatTensor of values in {0,1} of training split labels
            a [K x H x W] torch.FloatTensor of values in {0,1} of training boundary labels
        """
        no_bg_masks = graph.orig_masks[1:]
        labels_nhw = util_.convert_mask_HW_to_NHW(gt_labels, to_ignore=[])

        # Potential merges
        indices = util_.neighboring_mask_indices(no_bg_masks, neighbor_dist=1, reduction_factor=1)
        max_potential_merge_edges = indices.shape[0]

        num_merges = min(int(np.ceil(no_bg_masks.shape[0] / 3)), max_potential_merge_edges)
        success, train_graph, new_index_to_merge_mapping = randomly_merge_masks(
            graph, rgb_img_features, xyz, num_merges, self.config['padding_config'], 
            labels_nhw, return_merge_indices=True)
        # Note: train_graph has BG node, but new_index_to_merge_mapping assumes no BG nodes for graph/train_graph

        if not success:
            return False, None, None, None

        ### Compute labels ###
        K = train_graph.mask.shape[0]  # Note: K here includes background node
        new_H = self.config['padding_config']['new_H']
        new_W = self.config['padding_config']['new_W']

        # Initialize tensors to return
        split_labels = torch.zeros(K, dtype=torch.float32, device=constants.DEVICE)
        boundary_labels = torch.zeros((K, new_H, new_W), dtype=torch.float32, device=constants.DEVICE)

        # Crop/Resize, compute split/boundary labels
        for k, (i, j) in new_index_to_merge_mapping.items():

            # Split label
            split_labels[k + 1] = 1.  # + 1 for background

            # Boundary label
            resized_m0 = gc.crop_tensor_to_nchw(no_bg_masks[i],
                                                *train_graph.crop_indices[k + 1],  # + 1 for background
                                                (new_H,new_W),
                                                mode='nearest')[0, 0]  # [h, w]
            resized_m1 = gc.crop_tensor_to_nchw(no_bg_masks[j],
                                                *train_graph.crop_indices[k + 1],  # + 1 for background
                                                (new_H,new_W),
                                                mode='nearest')[0, 0]  # [h, w]

            temp = util_.mask_boundary_overlap(resized_m0.cpu().numpy(), resized_m1.cpu().numpy())
            boundary_labels[k + 1] = torch.from_numpy(temp).float().to(constants.DEVICE)  # + 1 for background

        return True, train_graph, split_labels, boundary_labels


    def train(self, num_epochs, data_loader):

        # Some stuff to keep track of
        batch_time = util_.AverageMeter()
        data_time = util_.AverageMeter()
        split_losses = util_.AverageMeter()
        boundary_losses = util_.AverageMeter()
        total_losses = util_.AverageMeter()
        end = time()

        # Training mode
        self.model_wrapper.train_mode()

        for epoch_iter in range(num_epochs):
            for batch in data_loader:

                if self.iter_num >= self.config['max_iters']:
                    print("Reached maximum number of iterations...")
                    self.save()
                    return

                # Send everything to GPU
                self.model_wrapper.send_batch_to_device(batch)

                # measure data loading time
                data_time.update(time() - end)

                rgb = batch['rgb'][0] # Shape: [3 x H x W]
                xyz = batch['xyz'][0] # Shape: [3 x H x W]
                seg_masks = batch['seg_masks'][0] # Shape: [H x W]
                labels = batch['foreground_labels'][0] # Shape: [H x W]
                H, W = labels.shape

                # Check that base graph has object IDs
                obj_ids = torch.unique(seg_masks)
                obj_ids = obj_ids[obj_ids >= constants.OBJECTS_LABEL]
                if obj_ids.shape[0] <= 1:
                    print(f"{obj_ids.shape[0]} graph nodes... discarding this example")
                    end = time()
                    continue

                # Build graph
                rgb_img_features = gc.extract_rgb_img_features(self.rn50_fpn, rgb)
                base_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, seg_masks, 
                                                             create_edge_indices=False,
                                                             padding_config=self.config['padding_config'])

                # Construct the training example
                success, train_graph, split_labels, boundary_labels = self.construct_training_example(
                    base_graph, rgb_img_features, xyz, labels)
                if not success:
                    print("Failed to construct training example...")
                    continue

                # Apply the model
                split_score_logits, boundary_logits = self.model_wrapper.model(train_graph)

                # Apply the loss
                split_loss = self.losses['split_score_loss'](split_score_logits, split_labels)
                boundary_loss = self.losses['boundary_loss'](boundary_logits, boundary_labels)
                loss = self.config['lambda_split'] * split_loss + \
                       self.config['lambda_boundary'] * boundary_loss


                ### Gradient descent ###
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                split_losses.update(split_loss.item(), split_labels.shape[0])
                boundary_losses.update(boundary_loss.item(), boundary_labels.shape[0])
                total_losses.update(loss.item(), split_labels.shape[0])

                # Record some information about this iteration
                batch_time.update(time() - end)
                end = time()

                # Record information every x iterations
                if self.iter_num % self.config['iter_collect'] == 0:
                    info = {'iter_num': self.iter_num,
                            'Batch Time': round(batch_time.avg, 3),
                            'Data Time': round(data_time.avg, 3),
                            'Split Loss': round(split_losses.avg, 7),
                            'Boundary Loss': round(boundary_losses.avg, 7),
                            'loss': round(total_losses.avg, 7),
                            }
                    self.infos[self.iter_num] = info

                    # Tensorboard stuff
                    self.tb_writer.add_scalar('Loss/Split Loss', info['Split Loss'], self.iter_num)
                    self.tb_writer.add_scalar('Loss/Boundary Loss', info['Boundary Loss'], self.iter_num)
                    self.tb_writer.add_scalar('Loss/Total', info['loss'], self.iter_num)
                    self.tb_writer.add_scalar('Time/per iter', info['Batch Time'], self.iter_num)
                    self.tb_writer.add_scalar('Time/data fetch', info['Data Time'], self.iter_num)

                    # Reset meters
                    batch_time = util_.AverageMeter()
                    data_time = util_.AverageMeter()
                    split_losses = util_.AverageMeter()
                    boundary_losses = util_.AverageMeter()
                    total_losses = util_.AverageMeter()
                    end = time()

                self.iter_num += 1

            self.epoch_num += 1

        self.save()


class DeleteNetTrainer(Trainer):

    def setup(self):

        # Losses
        self.losses = {
            'delete_loss' : nn.BCEWithLogitsLoss(),
        }

    def construct_training_example(self, graph, rgb_img_features, xyz, gt_labels):
        """DeleteNet training example construction.

        Args:
            graph: a torch_geometric.Data instance.
            rgb_img_features: an OrderedDict of ResNet50+FPN features. output of gc.extract_rgb_img_features
            xyz: a [3, H, W] torch.FloatTensor. Organized point cloud by backprojecting a depth map
            gt_labels: a [H, W] torch.FloatTensor of values in {0,1}. N_gt = #GT objects

        Returns:
            a Data instance
            a [N] torch.FloatTensor of delete labels
        """
        num_adds = torch.randint(self.config['perturbed_masks_min'],
                                 self.config['perturbed_masks_max']+1,
                                 size=[1]
                                )[0]
        success, train_graph = randomly_add_masks(graph, rgb_img_features, xyz, num_adds,
                                                  self.config['padding_config'], self.config)
        if not success:
            return False, None, None

        # Find corresponding GT label for each mask
        labels_nhw = util_.convert_mask_HW_to_NHW(gt_labels, to_ignore=[], to_keep=[0,1])
        mask_labels = util_.mask_corresponding_gt(train_graph.orig_masks, labels_nhw)
        delete_labels = (mask_labels < constants.OBJECTS_LABEL).float()  # < constants.OBJECTS_LABEL is background/table
        delete_labels[0] = 0.  # background node should never be deleted

        return True, train_graph, delete_labels

    def train(self, num_epochs, data_loader):

        # Some stuff to keep track of
        batch_time = util_.AverageMeter()
        data_time = util_.AverageMeter()
        total_losses = util_.AverageMeter()
        end = time()

        # Training mode
        self.model_wrapper.train_mode()

        for epoch_iter in range(num_epochs):
            for batch in data_loader:

                if self.iter_num >= self.config['max_iters']:
                    print("Reached maximum number of iterations...")
                    self.save()
                    return

                # Send everything to GPU
                self.model_wrapper.send_batch_to_device(batch)

                # measure data loading time
                data_time.update(time() - end)

                rgb = batch['rgb'][0] # Shape: [3 x H x W]
                xyz = batch['xyz'][0] # Shape: [3 x H x W]
                seg_masks = batch['seg_masks'][0] # Shape: [H x W]
                labels = batch['foreground_labels'][0] # Shape: [H x W]
                H, W = labels.shape

                # Check that base graph has object IDs
                obj_ids = torch.unique(seg_masks)
                obj_ids = obj_ids[obj_ids >= constants.OBJECTS_LABEL]
                # print(f"Number of nodes for base graph: {obj_ids.shape[0]}")
                if obj_ids.shape[0] == 0:
                    print(f"{obj_ids.shape[0]} graph nodes... discarding this example")
                    end = time()
                    continue

                # Build graph
                rgb_img_features = gc.extract_rgb_img_features(self.rn50_fpn, rgb)
                base_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, seg_masks, 
                                                             create_edge_indices=False,
                                                             padding_config=self.config['padding_config'])

                # Construct the training example
                success, train_graph, delete_labels = self.construct_training_example(
                    base_graph, rgb_img_features, xyz, labels)
                if not success:
                    print("Failed to construct training example...")
                    continue

                # Apply the model/loss
                out = self.model_wrapper.model(train_graph)  # [N]
                loss = self.losses['delete_loss'](out, delete_labels)


                ### Gradient descent ###
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                total_losses.update(loss.item(), delete_labels.shape[0])

                # Record some information about this iteration
                batch_time.update(time() - end)
                end = time()

                # Record information every x iterations
                if self.iter_num % self.config['iter_collect'] == 0:
                    info = {'iter_num': self.iter_num,
                            'Batch Time': round(batch_time.avg, 3),
                            'Data Time': round(data_time.avg, 3),
                            'loss': round(total_losses.avg, 7),
                            }
                    self.infos[self.iter_num] = info

                    # Tensorboard stuff
                    self.tb_writer.add_scalar('Loss/Total', info['loss'], self.iter_num)
                    self.tb_writer.add_scalar('Time/per iter', info['Batch Time'], self.iter_num)
                    self.tb_writer.add_scalar('Time/data fetch', info['Data Time'], self.iter_num)

                    # Reset meters
                    batch_time = util_.AverageMeter()
                    data_time = util_.AverageMeter()
                    total_losses = util_.AverageMeter()
                    end = time()

                self.iter_num += 1

            self.epoch_num += 1

        self.save()


class JointSplitNetDeleteNetTrainer:

    def __init__(self, 
                 splitnet_wrapper, splitnet_trainer,
                 deletenet_wrapper, deletenet_trainer,
                 load_config=None,
                ):

        self.splitnet_wrapper = splitnet_wrapper
        self.deletenet_wrapper = deletenet_wrapper
        self.splitnet_trainer = splitnet_trainer
        self.deletenet_trainer = deletenet_trainer

        # Ensure ResNet50+FPN model is shared across both trainers
        assert self.splitnet_trainer.rn50_fpn is self.deletenet_trainer.rn50_fpn
        self.rn50_fpn = self.splitnet_trainer.rn50_fpn

        # Make sure splitnet/deletenet trainers agree on a few things
        required_agrees = ['trainable_layer_names',
                           'lr',
                           'max_iters',
                           'tb_directory',
                           'iter_collect',
                           ]
        for key in required_agrees:
            assert self.splitnet_trainer.config[key] == self.deletenet_trainer.config[key], (
                   f'{key} does not agree... {self.splitnet_trainer.config[key], self.deletenet_trainer.config[key]}')

        # Share NodeEncoder CNNs
        self.deletenet_wrapper.model.node_encoder.encoders = self.splitnet_wrapper.model.node_encoder.encoders

        # Initialize optimizer
        model_params = dict(self.splitnet_wrapper.model.named_parameters())
        model_params.update(dict(self.deletenet_wrapper.model.named_parameters()))
        model_params = list(model_params.values())
        if len(self.splitnet_trainer.config['trainable_layer_names']) > 0:
            model_params += [p for p in self.rn50_fpn.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(model_params, lr=self.splitnet_trainer.config['lr'])

        # Make sure optimizer has the correct set of params (wrt SplitNet/DeleteNet)
        for param in self.deletenet_wrapper.model.parameters():
            assert any([param is p for p in model_params])
        for param in self.splitnet_wrapper.model.parameters():
            assert any([param is p for p in model_params])

        # Tensorboard stuff
        self.tb_directory = self.splitnet_trainer.config['tb_directory']  # should be same as deletenet_trainer.
        self.iter_collect = self.splitnet_trainer.config['iter_collect']
        self.tb_writer = SummaryWriter(self.tb_directory, flush_secs=10)

        # Misc stuff
        self.infos = dict()
        self.max_iters = self.splitnet_trainer.config['max_iters']
        self.iter_num = 0
        self.epoch_num = 0

        # Load pretrained models for fine-tuning.
        if load_config:
            self.load(opt_filename=load_config['opt_filename'],
                      splitnet_wrapper_filename=load_config['splitnet_wrapper_filename'],
                      deletenet_wrapper_filename=load_config['deletenet_wrapper_filename'],
                      rn50_fpn_filename=load_config['rn50_fpn_filename'],
                     )
            

    def train(self, num_epochs, data_loader):

        # Some stuff to keep track of
        deletenet_batch_time = util_.AverageMeter()
        deletenet_losses = util_.AverageMeter()
        splitnet_batch_time = util_.AverageMeter()
        split_losses = util_.AverageMeter()
        boundary_losses = util_.AverageMeter()
        splitnet_losses = util_.AverageMeter()
        optimizer_time = util_.AverageMeter()

        end = time()

        # Training mode
        self.splitnet_wrapper.train_mode()
        self.deletenet_wrapper.train_mode()

        deletenet_iter = True  # Start with DeleteNet iteration
        for epoch_iter in range(num_epochs):
            for batch in data_loader:

                if self.iter_num >= self.max_iters:
                    print("Reached maximum number of iterations...")
                    self.save()
                    return

                self.splitnet_wrapper.send_batch_to_device(batch)  # Send everything to GPU

                rgb = batch['rgb'][0] # Shape: [3 x H x W]
                xyz = batch['xyz'][0] # Shape: [3 x H x W]
                seg_masks = batch['seg_masks'][0] # Shape: [H x W]
                labels = batch['foreground_labels'][0] # Shape: [H x W]

                # Check that base graph has object IDs
                obj_ids = torch.unique(seg_masks)
                obj_ids = obj_ids[obj_ids >= constants.OBJECTS_LABEL]
                if obj_ids.shape[0] == 0:
                    print(f"{obj_ids.shape[0]} graph nodes... discarding this example")
                    end = time()
                    continue

                # Build graph
                rgb_img_features = gc.extract_rgb_img_features(self.rn50_fpn, rgb)
                base_graph = gc.construct_segmentation_graph(rgb_img_features, xyz, seg_masks, 
                                                             create_edge_indices=False,
                                                             padding_config=self.splitnet_trainer.config['padding_config'])

                if deletenet_iter:

                    # Construct the training example
                    success, train_graph, delete_labels = self.deletenet_trainer.construct_training_example(
                        base_graph, rgb_img_features, xyz, labels)
                    if not success:
                        print("Failed to construct DeleteNet training example...")
                        continue

                    # Apply the DeleteNet model/loss
                    out = self.deletenet_wrapper.model(train_graph)  # Shape: [N]
                    deletenet_loss = self.deletenet_trainer.losses['delete_loss'](
                        out.unsqueeze(0), delete_labels.unsqueeze(0))

                    # measure accuracy and record loss
                    deletenet_losses.update(deletenet_loss.item(), delete_labels.shape[0])

                    # Record some information about this iteration
                    deletenet_batch_time.update(time() - end)
                    end = time()

                    deletenet_iter = False

                else:  # SplitNet training iteration

                    # Construct the training example
                    success, train_graph, split_labels, boundary_labels = self.splitnet_trainer.construct_training_example(
                        base_graph, rgb_img_features, xyz, labels)
                    if not success:
                        print("Failed to construct SplitNet training example...")
                        continue

                    # Apply the model
                    split_score_logits, boundary_logits = self.splitnet_wrapper.model(train_graph)

                    # Apply the loss
                    split_loss = self.splitnet_trainer.losses['split_score_loss'](split_score_logits, split_labels)
                    boundary_loss = self.splitnet_trainer.losses['boundary_loss'](boundary_logits, boundary_labels)
                    splitnet_loss = self.splitnet_trainer.config['lambda_split'] * split_loss + \
                                    self.splitnet_trainer.config['lambda_boundary'] * boundary_loss                    

                    # measure accuracy and record loss
                    split_losses.update(split_loss.item(), split_labels.shape[0])
                    boundary_losses.update(boundary_loss.item(), split_labels.shape[0])
                    splitnet_losses.update(splitnet_loss.item(), split_labels.shape[0])

                    # Record some information about this iteration
                    splitnet_batch_time.update(time() - end)
                    end = time()



                    ### Gradient descent ###
                    total_loss = deletenet_loss + splitnet_loss
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    optimizer_time.update(time() - end)
                    end = time()

                    # Record information every x iterations
                    if self.iter_num % self.iter_collect == 0:
                        info = {'iter_num': self.iter_num,

                                'DeleteNet Loss': round(deletenet_losses.avg, 7),
                                'DeleteNet Batch Time': round(deletenet_batch_time.avg, 3),

                                'Split Loss': round(split_losses.avg, 7),
                                'Boundary Loss': round(boundary_losses.avg, 7),
                                'SplitNet Loss': round(splitnet_losses.avg, 7),
                                'SplitNet Batch Time': round(splitnet_batch_time.avg, 3),

                                'Optimizer Time': round(optimizer_time.avg, 3),
                                }
                        self.infos[self.iter_num] = info

                        # Tensorboard stuff
                        self.tb_writer.add_scalar('Loss/DeleteNet/Total', info['DeleteNet Loss'], self.iter_num)
                        self.tb_writer.add_scalar('Loss/SplitNet/Total', info['SplitNet Loss'], self.iter_num)
                        self.tb_writer.add_scalar('Loss/SplitNet/Split Loss', info['Split Loss'], self.iter_num)
                        self.tb_writer.add_scalar('Loss/SplitNet/Boundary Loss', info['Boundary Loss'], self.iter_num)
                        self.tb_writer.add_scalar('Time/DeleteNet batch', info['DeleteNet Batch Time'], self.iter_num)
                        self.tb_writer.add_scalar('Time/SplitNet batch', info['SplitNet Batch Time'], self.iter_num)
                        self.tb_writer.add_scalar('Time/Optimizer', info['Optimizer Time'], self.iter_num)

                        # Reset meters
                        deletenet_batch_time = util_.AverageMeter()
                        deletenet_losses = util_.AverageMeter()
                        splitnet_batch_time = util_.AverageMeter()
                        split_losses = util_.AverageMeter()
                        boundary_losses = util_.AverageMeter()
                        splitnet_losses = util_.AverageMeter()
                        optimizer_time = util_.AverageMeter()
                        end = time()

                    deletenet_iter = True
                    self.iter_num += 1

            self.epoch_num += 1
            self.save()


    def save(self, save_dir=None):

        # Save optimizer stuff
        checkpoint = {
            'iter_num' : self.iter_num,
            'epoch_num' : self.epoch_num,
            'infos' : self.infos,
        }

        checkpoint['optimizer'] = self.optimizer.state_dict()

        if save_dir is None:
            save_dir = self.tb_directory

        filename = os.path.join(save_dir,
                                (self.__class__.__name__ + '_' +
                                 self.splitnet_wrapper.__class__.__name__ +
                                 self.deletenet_wrapper.__class__.__name__ +
                                 '_iter' + str(self.iter_num) +
                                 '_checkpoint.pth')
                                )
        torch.save(checkpoint, filename)

        # Save SplitNet/DeleteNet model wrappers
        splitnet_wrapper_filename = os.path.join(save_dir,
                                                 (self.splitnet_wrapper.__class__.__name__ +
                                                  '_iter' + str(self.iter_num) +
                                                  '_checkpoint.pth')
                                                 )
        self.splitnet_wrapper.save(splitnet_wrapper_filename)
        deletenet_wrapper_filename = os.path.join(save_dir,
                                                  (self.deletenet_wrapper.__class__.__name__ +
                                                   '_iter' + str(self.iter_num) +
                                                   '_checkpoint.pth')
                                                  )
        self.deletenet_wrapper.save(deletenet_wrapper_filename)

        # Save ResNet50 FPN
        if len(self.splitnet_trainer.config['trainable_layer_names']) > 0:
            checkpoint = {'model' : self.rn50_fpn.state_dict()}
            filename = os.path.join(save_dir,
                                    (self.rn50_fpn.__class__.__name__ +
                                     '_iter' + str(self.iter_num) +
                                     '_checkpoint.pth')
                                    )
            torch.save(checkpoint, filename)


    def load(self,
             opt_filename=None,
             splitnet_wrapper_filename=None,
             deletenet_wrapper_filename=None,
             rn50_fpn_filename=None):

        # Load optimizer stuff
        if opt_filename is not None:
            checkpoint = torch.load(opt_filename)

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded optimizer")

            self.iter_num = checkpoint['iter_num']
            self.epoch_num = checkpoint['epoch_num']
            self.infos = checkpoint['infos']

        # Load model stuff
        if splitnet_wrapper_filename is not None:
            self.splitnet_wrapper.load(splitnet_wrapper_filename)
        if deletenet_wrapper_filename is not None:
            self.deletenet_wrapper.load(deletenet_wrapper_filename)

        # Load ResNet50+FPN file (if modality tuning)
        if rn50_fpn_filename is not None and \
           len(self.splitnet_trainer.config['trainable_layer_names']) > 0:
            checkpoint = torch.load(rn50_fpn_filename)
            self.rn50_fpn.load_state_dict(checkpoint['model'])
            print("Loaded ResNet50+FPN")
