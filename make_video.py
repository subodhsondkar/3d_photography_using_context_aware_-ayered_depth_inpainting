import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
import sys
from mesh import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
import json
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering


def make_video(params_file):
    print("Making Video...")
    with open(params_file) as f:
        params = json.load(f)

    os.makedirs(params['mesh_folder'], exist_ok=True)
    os.makedirs(params['video_folder'], exist_ok=True)
    os.makedirs(params['depth_folder'], exist_ok=True)
    sample_list = get_MiDaS_samples(params, params['specific'])
    normal_canvas, all_canvas = None, None

    device = "cuda"

    depth = None
    try:
        sample = sample_list[0]
    except:
        sample = sample_list
    # print("Current Source ==> ", sample['src_pair_name'])
    mesh_fi = os.path.join(params['mesh_folder'], sample['src_pair_name'] +'.ply')
    image = imageio.imread(sample['ref_img_fi'])

    # print(f"Running depth extraction at {time.time()}")
    if params['require_midas'] is True:
        run_depth([sample['ref_img_fi']], params['src_dir'], params['depth_folder'],
                params['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=640)
    if 'npy' in params['depth_format']:
        params['output_h'], params['output_w'] = np.load(sample['depth_fi']).shape[:2]
    else:
        params['output_h'], params['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
    frac = params['longer_side_len'] / max(params['output_h'], params['output_w'])
    params['output_h'], params['output_w'] = int(params['output_h'] * frac), int(params['output_w'] * frac)
    params['original_h'], params['original_w'] = params['output_h'], params['output_w']
    if image.ndim == 2:
        image = image[..., None].repeat(3, -1)
    if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
        params['gray_image'] = True
    else:
        params['gray_image'] = False
    image = cv2.resize(image, (params['output_w'], params['output_h']), interpolation=cv2.INTER_AREA)
    depth = read_MiDaS_depth(sample['depth_fi'], 3.0, params['output_h'], params['output_w'])
    mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
    if not(params['load_ply'] is True and os.path.exists(mesh_fi)):
        vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), params, num_iter=params['sparse_iter'], spdb=False)
        depth = vis_depths[-1]
        model = None
        torch.cuda.empty_cache()

        depth_edge_model = Inpaint_Edge_Net(init_weights=True)
        depth_edge_weight = torch.load(params['depth_edge_model_ckpt'],
                                    map_location=torch.device(device))
        depth_edge_model.load_state_dict(depth_edge_weight)
        depth_edge_model = depth_edge_model.to(device)
        depth_edge_model.eval()

        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load(params['depth_feat_model_ckpt'],
                                    map_location=torch.device(device))
        depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
        depth_feat_model = depth_feat_model.to(device)
        depth_feat_model.eval()
        depth_feat_model = depth_feat_model.to(device)

        rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load(params['rgb_feat_model_ckpt'],
                                    map_location=torch.device(device))
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model.eval()
        rgb_model = rgb_model.to(device)
        graph = None

        rt_info = write_ply("", image, depth, sample['int_mtx'], mesh_fi, params, rgb_model, depth_edge_model, depth_edge_model, depth_feat_model)

        if rt_info is False:
            return
        rgb_model = None
        color_feat_model = None
        depth_edge_model = None
        depth_feat_model = None
        torch.cuda.empty_cache()
    if params['save_ply'] is True or params['load_ply'] is True:
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
    else:
        verts, colors, faces, Height, Width, hFov, vFov = rt_info

    videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
    top = (params.get('original_h') // 2 - sample['int_mtx'][1, 2] * params['output_h'])
    left = (params.get('original_w') // 2 - sample['int_mtx'][0, 2] * params['output_w'])
    down, right = top + params['output_h'], left + params['output_w']
    border = [int(xx) for xx in [top, down, left, right]]
    normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                        copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(sample['ref_pose']), copy.deepcopy(params['video_folder']),
                        image.copy(), copy.deepcopy(sample['int_mtx']), params, image,
                        videos_poses, video_basename, params.get('original_h'), params.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                        mean_loc_depth=mean_loc_depth)
    print("Done!")
if __name__ == '__main__':
    make_video('params.json')