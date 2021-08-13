import os
from collections import OrderedDict
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU

from . import constants


"""Config schemas

This file loads config files and computes some extra stuff.
    For example, if a NodeEncoder schema uses CNNs for the RGB/Depth
    pathways but uses a LinearEncoder for the Mask pathway, it will
    automatically calculate what the fusion module input dimension
    should be.


SplitNet, DeleteNet, and SGS-Net each have separate configs. 
The configs are largely similar. They each (potentially) have:

- node_encoder_config:
    - {rgb,depth,mask}_encoder_config
    - These describe the CNN and/or linear encoders applied to 
        each input. Note that in the paper, no linear encoders
        are used. They are typically all CNN encoders.
    - fusion_module_config (SGS-Net only)
- decoder_config (SplitNet only)
- bg_fusion_module_config (DeleteNet only)
- gn_layer_config (SGS-Net only)

Note that the CNN encoders for the NodeEncoder are shared amongst
  SplitNet/DeleteNet/SGS-Net.
"""


def dictify(dict_, dict_type=dict):
    """Turn nested dicts into nested dict_type dicts.

    E.g. turn OrderedDicts into dicts, or dicts into OrderedDicts.
    """
    
    if not isinstance(dict_, dict):
        return dict_
    
    new_dict = dict_type()
    for key in dict_:
        if isinstance(dict_[key], dict):
            new_dict[key] = dictify(dict_[key], dict_type=dict_type)
        elif isinstance(dict_[key], list):
            new_dict[key] = [dictify(x, dict_type=dict_type) for x in dict_[key]]
        else:
            new_dict[key] = dict_[key]
            
    return new_dict


def process_node_encoder_config(node_encoder_cfg,
                                img_size=(64, 64),
                                CNN_reduction_factor=None):
    """Update node encoder config dictionary in place."""

    # For the CNN encoders, compute linear encoder (on top of CNN) input if specified
    for key in node_encoder_cfg.keys():
        if 'encoder_config' not in key:
            continue

        if (node_encoder_cfg[key]['type'] == 'cnn' and
            'linear_encoder_config' in node_encoder_cfg[key]):
            reduced_img_size = (np.array(img_size) / CNN_reduction_factor / node_encoder_cfg[key]['avg_pool_kernel_size'])
            input_dim = int(np.prod(reduced_img_size) * node_encoder_cfg[key]['output_channels'])
            node_encoder_cfg[key]['linear_encoder_config']['input_dim'] = input_dim

    # If fusion module is specified, compute input dimension (fusing the features from each encoder)
    if 'fusion_module_config' in node_encoder_cfg:
        fm_inc = 0
        for key in node_encoder_cfg:
            if 'encoder_config' not in key:
                continue
            if node_encoder_cfg[key]['type'] == 'linear':
                fm_inc += node_encoder_cfg[key]['output_dim']
            elif node_encoder_cfg[key]['type'] == 'cnn':  # Note, linear fusion module assumes CNNs have additional linear module on top
                fm_inc += node_encoder_cfg[key]['linear_encoder_config']['output_dim']
        node_encoder_cfg['fusion_module_config']['input_dim'] = fm_inc



def get_splitnet_config(cfg_filename):
    """Get SplitNet config from file, perform additional computation."""

    # First, load the YAML file
    with open(cfg_filename, 'r') as f:
        cfg = yaml.load(f)

    # Process node encoder config
    node_encoder_cfg = cfg['node_encoder_config']
    process_node_encoder_config(node_encoder_cfg,
                                img_size=cfg['img_size'])

    # Compute encoder output channels/dims for decoder input
    encoder_output_channels = dict()
    encoder_output_dims = dict()
    for k in node_encoder_cfg.keys():
        if 'encoder_config' in k:
            if node_encoder_cfg[k]['type'] == 'cnn':
                encoder_output_channels[k] = node_encoder_cfg[k]['output_channels']
            if node_encoder_cfg[k]['type'] == 'linear':
                encoder_output_dims[k] = node_encoder_cfg[k]['output_dim']
                
    cfg['decoder_config'].update({
        'encoder_output_channels' : encoder_output_channels,
        'encoder_output_dims' : encoder_output_dims,
        'img_size' : cfg['img_size'],
    })

    return dictify(cfg, dict_type=OrderedDict)


def get_splitnet_train_config(cfg_filename):
    """Get SplitNet training config from file, perform additional computation."""

    # First, load the YAML file
    with open(cfg_filename, 'r') as f:
        train_cfg = yaml.load(f)

    # Extra things to compute
    train_cfg['tb_directory'] = os.path.join(constants.BASE_TENSORBOARD_DIR, train_cfg['tb_directory'])

    iter_num = train_cfg['iter_num']
    train_cfg['opt_filename'] = os.path.join(train_cfg['tb_directory'],
                                             f'SplitNetTrainer_SplitNetWrapper_iter{iter_num}_checkpoint.pth')
    train_cfg['model_filename'] = os.path.join(train_cfg['tb_directory'],
                                               f'SplitNetWrapper_iter{iter_num}_checkpoint.pth')
    train_cfg['rn50_fpn_filename'] = os.path.join(train_cfg['tb_directory'],
                                                  f'BackboneWithFPN_iter{iter_num}_checkpoint.pth')

    return train_cfg


def get_deletenet_config(cfg_filename):
    """Get DeleteNet config from file, perform additional computation."""

    # First, load the YAML file
    with open(cfg_filename, 'r') as f:
        cfg = yaml.load(f)

    # Process node encoder config
    node_encoder_cfg = cfg['node_encoder_config']
    process_node_encoder_config(node_encoder_cfg,
                                img_size=cfg['img_size'],
                                CNN_reduction_factor=cfg['CNN_reduction_factor'])

    bg_fm_inc = 0
    for key in node_encoder_cfg:
        if 'encoder_config' not in key:
            continue
        if node_encoder_cfg[key]['type'] == 'linear':
            bg_fm_inc += node_encoder_cfg[key]['output_dim']
        elif node_encoder_cfg[key]['type'] == 'cnn':  # Note, linear fusion module assumes CNNs have additional linear module on top
            bg_fm_inc += node_encoder_cfg[key]['linear_encoder_config']['output_dim']
    cfg['bg_fusion_module_config']['input_dim'] = bg_fm_inc

    return dictify(cfg, dict_type=OrderedDict)


def get_deletenet_train_config(cfg_filename):
    """Get DeleteNet training config from file, perform additional computation."""

    # First, load the YAML file
    with open(cfg_filename, 'r') as f:
        train_cfg = yaml.load(f)

    # Extra things to compute
    train_cfg['tb_directory'] = os.path.join(constants.BASE_TENSORBOARD_DIR, train_cfg['tb_directory'])

    iter_num = train_cfg['iter_num']
    train_cfg['opt_filename'] = os.path.join(train_cfg['tb_directory'],
                                             f'DeleteNetTrainer_DeleteNetWrapper_iter{iter_num}_checkpoint.pth')
    train_cfg['model_filename'] = os.path.join(train_cfg['tb_directory'],
                                               f'DeleteNetWrapper_iter{iter_num}_checkpoint.pth')

    return train_cfg


def get_sgsnet_config(cfg_filename):
    """Get SGSNet config from file, perform additional computation."""

    # First, load the YAML file
    with open(cfg_filename, 'r') as f:
        cfg = yaml.load(f)

    # Process node encoder config
    node_encoder_cfg = cfg['node_encoder_config']
    process_node_encoder_config(node_encoder_cfg,
                                img_size=cfg['img_size'],
                                CNN_reduction_factor=cfg['CNN_reduction_factor'])

    # GraphNet layers
    cfg['layer_config'] = []
    for i in range(cfg['num_gn_layers']):
        gn_layer_config = cfg['gn_layer_config'].copy()
        if i == 0:
            gn_layer_config['node_input_channels'] = node_encoder_cfg['fusion_module_config']['output_dim']
            gn_layer_config['edge_input_channels'] = node_encoder_cfg['fusion_module_config']['output_dim']
        else:
            gn_layer_config['node_input_channels'] = cfg['layer_config'][-1]['node_output_channels']
            gn_layer_config['edge_input_channels'] = cfg['layer_config'][-1]['edge_output_channels']
        cfg['layer_config'].append(gn_layer_config)

    # GraphNet Output layer
    cfg['gn_output_layer']['input_dim'] = (cfg['layer_config'][-1]['node_output_channels'] +
                                           cfg['layer_config'][-1]['edge_output_channels'])

    return dictify(cfg, dict_type=OrderedDict)


def get_sgsnet_train_config(cfg_filename):
    """Get SGSNet training config from file, perform additional computation."""

    # First, load the YAML file
    with open(cfg_filename, 'r') as f:
        train_cfg = yaml.load(f)

    # Extra things to compute
    train_cfg['tb_directory'] = os.path.join(constants.BASE_TENSORBOARD_DIR, train_cfg['tb_directory'])

    # Get filenames for pretrained ResNet50+FPN and SplitNet (to get encoders)
    train_cfg['rn50_fpn_filename'] = os.path.join(
        constants.BASE_TENSORBOARD_DIR,
        train_cfg['pretrained_tb_path'],
        f'BackboneWithFPN_iter{train_cfg["pretrained_iter_num"]}_checkpoint.pth')
    train_cfg['splitnet_filename'] = os.path.join(
        constants.BASE_TENSORBOARD_DIR,
        train_cfg['pretrained_tb_path'],
        f'SplitNetWrapper_iter{train_cfg["pretrained_iter_num"]}_checkpoint.pth')

    iter_num = train_cfg['iter_num']
    train_cfg['opt_filename'] = os.path.join(train_cfg['tb_directory'],
                                             f'SGSNetTrainer_SGSNetWrapper_iter{iter_num}_checkpoint.pth')
    train_cfg['model_filename'] = os.path.join(train_cfg['tb_directory'],
                                               f'SGSNetWrapper_iter{iter_num}_checkpoint.pth')

    return train_cfg

