import abc
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from diffusers.utils.constants import USE_PEFT_BACKEND
from matplotlib import pyplot as plt
from torch import einsum
from einops import rearrange
import math
import numpy as np
import pandas as pd
import scipy.io as sio
import os
from PIL import Image
from torchvision.transforms import PILToTensor, ToPILImage

from matplotlib.colors import ListedColormap
import json
from glob import glob

def get_dataset_info(args, split):
    if args.dataset == 'pascal':
        data_dir = args.dataset_path
        categories = sorted(os.listdir(os.path.join(data_dir, 'Annotations')))
    elif args.dataset == 'ap10k':
        data_dir = args.dataset_path
        categories = []
        subfolders = os.listdir(os.path.join(data_dir, 'ImageAnnotation'))
        # Handle AP10K_EVAL test settings
        if args.AP10K_EVAL_SUBSET == 'intra-species':
            categories = [folder for subfolder in subfolders for folder in os.listdir(os.path.join(data_dir, 'ImageAnnotation', subfolder))]
        elif args.AP10K_EVAL_SUBSET == 'cross-species':
            categories = [subfolder for subfolder in subfolders if len(os.listdir(os.path.join(data_dir, 'ImageAnnotation', subfolder))) > 1]
            split += '_cross_species'
        elif args.AP10K_EVAL_SUBSET == 'cross-family':
            categories = ['all']
            split += '_cross_family'
        categories = sorted(categories)
        if split == 'val':
            # remove category "king cheetah" from categories, since it is not present in the validation set
            categories.remove('king cheetah')
    else: # SPair
        data_dir = args.dataset_path
        categories = sorted(os.listdir(os.path.join(data_dir, 'ImageAnnotation')))

    return data_dir, categories, split

def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale

def preprocess_kps_less_1024(kps, img_width, img_height, max_size=1024):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    
    new_w = img_width
    new_h = img_height
    scale = 1.0
    if max(img_width, img_height)>max_size:
        scale = max_size / max(img_width, img_height)
        kps[:, [0, 1]] *= scale
        if img_height < img_width:
            new_w = max_size
            new_h = int(np.around(max_size * img_height / img_width))
        else:
            new_w = int(np.around(max_size * img_width / img_height))
            new_h = max_size
    return kps, new_w, new_h, scale

def draw_correspondences_gathered(points1, points2, image1, image2):
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: a figure of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)
    
    # plot a subfigure put image1 in the top, image2 in the bottom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.025)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)

    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)

    return fig

def resize(img, target_res=224, resize=True, to_pil=True, edge=False, one_pading=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        if one_pading:
            canvas = np.ones([target_res, target_res, 3], dtype=np.uint8)*255
            # print(canvas[0])
        else:
            canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

def load_ap10k_data(path, size=768, category='cat', split='test', subsample=0):
    np.random.seed(42)
    pairs = sorted(list(glob(f'{path}/PairAnnotation/{split}/*:{category}.json')))
    # if subsample is not None and subsample > 0:
    #     pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    files = []
    kps = []
    thresholds = []
    data_category = []
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        source_json_path = data["src_json_path"]
        target_json_path = data["trg_json_path"]
        
        src_imname = ((source_json_path).split(".")[0]).replace('ImageAnnotation', 'JPEGImages').replace('dataset/ap-10k/','')
        trg_imname = ((target_json_path).split(".")[0]).replace('ImageAnnotation', 'JPEGImages').replace('dataset/ap-10k/','')
        
        
        source_json_path = "/mnt/nvme0n1/chaofan/" + source_json_path
        target_json_path = "/mnt/nvme0n1/chaofan/" + target_json_path
        
        src_img_path = source_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
        trg_img_path = target_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')

        with open(source_json_path) as f:
            src_file = json.load(f)
        with open(target_json_path) as f:
            trg_file = json.load(f)
            
        source_bbox = np.asarray(src_file["bbox"])  # l t w h
        target_bbox = np.asarray(trg_file["bbox"])
        
        source_size = np.array([src_file["width"], src_file["height"]])  # (W, H)
        target_size = np.array([trg_file["width"], trg_file["height"]])  # (W, H)

        # print(source_raw_kps.shape)
        source_kps = torch.tensor(src_file["keypoints"]).view(-1, 3).float()
        source_kps[:,-1] /= 2
        # source_kps, source_new_w, source_new_h, src_scale = preprocess_kps_less_1024(source_kps, source_size[0], source_size[1])
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)

        
        
        target_kps = torch.tensor(trg_file["keypoints"]).view(-1, 3).float()
        target_kps[:,-1] /= 2
        # target_kps, target_new_w, target_new_h, trg_scale = preprocess_kps_less_1024(target_kps, target_size[0], target_size[1])
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        
        # The source thresholds aren't actually used to evaluate PCK on SPair-71K, but for completeness
        # they are computed as well:
        # thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]))
        if ('test' in split) or ('val' in split):
            thresholds.append(max(target_bbox[3], target_bbox[2])*trg_scale)
        elif 'trn' in split:
            thresholds.append(max(source_bbox[3], source_bbox[2])*src_scale)
            thresholds.append(max(target_bbox[3], target_bbox[2])*trg_scale)

        # kps.append(source_kps)
        # kps.append(target_kps)
        files.append(src_imname)
        files.append(trg_imname)
        # data_pair = {""}
        
        vis = source_kps[:, 2] * target_kps[:, 2] > 0
        
        data_pair = {"src_imsize":[size, size],
                     "trg_imsize":[size, size],
                     "src_imname":src_imname,
                     "trg_imname":trg_imname,
                     "threshold":max(target_bbox[3], target_bbox[2])*trg_scale,
                     "src_kps":(source_kps[vis].to(torch.int32)).numpy().tolist(),
                     "trg_kps":(target_kps[vis].to(torch.int32)).numpy().tolist()}
        
        data_category.append(data_pair)
        

    # kps = torch.stack(kps)
    # used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    # kps = kps[:, used_kps, :]
    # print(f'Final number of used key points: {kps.size(1)}')
    files = list(set(files))
    return files, data_category

# SPair-71K

def load_spair_data(path, size=768, category='cat', split='test', subsample=None):
    # print(size)
    np.random.seed(42)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    files = []
    thresholds = []
    data_category = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    for pair in pairs:
        source_kps = torch.zeros(num_kps, 3)
        target_kps = torch.zeros(num_kps, 3)
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        
        # src_imname = ((source_json_path).split(".")[0]).replace('ImageAnnotation', 'JPEGImages')
        # trg_imname = ((target_json_path).split(".")[0]).replace('ImageAnnotation', 'JPEGImages')
        
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        
        src_imname = (f'JPEGImages/{category}/{data["src_imname"]}').split(".")[0]
        trg_imname = (f'JPEGImages/{category}/{data["trg_imname"]}').split(".")[0]
        
        source_json_name = source_fn.replace('JPEGImages','ImageAnnotation').replace('jpg','json')
        target_json_name = target_fn.replace('JPEGImages','ImageAnnotation').replace('jpg','json')
        source_bbox = np.asarray(data["src_bndbox"])    # (x1, y1, x2, y2)
        target_bbox = np.asarray(data["trg_bndbox"])
        with open(source_json_name) as f:
            file = json.load(f)
            kpts_src = file['kps']
        with open(target_json_name) as f:
            file = json.load(f)
            kpts_trg = file['kps']

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        for i in range(30):
            point = kpts_src[str(i)]
            if point is None:
                source_kps[i, :3] = 0
            else:
                source_kps[i, :2] = torch.Tensor(point).float()  # set x and y
                source_kps[i, 2] = 1
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        
        for i in range(30):
            point = kpts_trg[str(i)]
            if point is None:
                target_kps[i, :3] = 0
            else:
                target_kps[i, :2] = torch.Tensor(point).float()
                target_kps[i, 2] = 1
        # target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        # target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        if split == 'test' or split == 'val':
            thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])*trg_scale)
        elif split == 'trn':
            thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0])*src_scale)
            thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])*trg_scale)

        files.append(src_imname)
        files.append(trg_imname)
        
        
        
        vis = source_kps[:, 2] * target_kps[:, 2] > 0
        
        data_pair = {"src_imsize":[size, size],
                     "trg_imsize":[size, size],
                     "src_imname":src_imname,
                     "trg_imname":trg_imname,
                     "threshold":max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])*trg_scale,
                     "src_kps":(source_kps[vis].to(torch.int32)).numpy().tolist(),
                     "trg_kps":(target_kps[vis].to(torch.int32)).numpy().tolist()}
        
        data_category.append(data_pair)
        
    files = list(set(files))
    return files, data_category

# Pascal

def read_mat(path, obj_name):
    r"""Reads specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj

def process_kps_pascal(kps):
    # Step 1: Reshape the array to (20, 2) by adding nan values
    num_pad_rows = 20 - kps.shape[0]
    if num_pad_rows > 0:
        pad_values = np.full((num_pad_rows, 2), np.nan)
        kps = np.vstack((kps, pad_values))
        
    # Step 2: Reshape the array to (20, 3) 
    # Add an extra column: set to 1 if the row does not contain nan, 0 otherwise
    last_col = np.isnan(kps).any(axis=1)
    last_col = np.where(last_col, 0, 1)
    kps = np.column_stack((kps, last_col))

    # Step 3: Replace rows with nan values to all 0's
    mask = np.isnan(kps).any(axis=1)
    kps[mask] = 0

    return torch.tensor(kps).float()

def load_pascal_data(path="data/PF-dataset-PASCAL", size=256, category='cat', split='test', subsample=None):
    
    def get_points(point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1,20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    
    np.random.seed(42)
    files = []
    data_category = []
    kps = []
    test_data = pd.read_csv(f'{path}/{split}_pairs_pf_pascal.csv')
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:,2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    # logger.info(f'Number of Pairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id,:]
    src_img_names = np.array(subset_pairs.iloc[:,0])
    trg_img_names = np.array(subset_pairs.iloc[:,1])
    # print(src_img_names.shape, trg_img_names.shape)
    if not split.startswith('train'):
        point_A_coords = subset_pairs.iloc[:,3:5]
        point_B_coords = subset_pairs.iloc[:,5:]
    # print(point_A_coords.shape, point_B_coords.shape)
    for i in range(len(src_img_names)):
        src_fn= f'{path}/../{src_img_names[i]}'
        trg_fn= f'{path}/../{trg_img_names[i]}'
        
        src_imname = (f'JPEGImages/{src_img_names[i]}').split(".")[0]
        trg_imname = (f'JPEGImages/{trg_img_names[i]}').split(".")[0]
        
        src_size=Image.open(src_fn).size
        trg_size=Image.open(trg_fn).size

        if not split.startswith('train'):
            point_coords_src = get_points(point_A_coords, i).transpose(1,0)
            point_coords_trg = get_points(point_B_coords, i).transpose(1,0)
        else:
            src_anns = os.path.join(path, 'Annotations', category,
                                    os.path.basename(src_fn))[:-4] + '.mat'
            trg_anns = os.path.join(path, 'Annotations', category,
                                    os.path.basename(trg_fn))[:-4] + '.mat'
            point_coords_src = process_kps_pascal(read_mat(src_anns, 'kps'))
            point_coords_trg = process_kps_pascal(read_mat(trg_anns, 'kps'))

        # print(src_size)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(point_coords_src, src_size[0], src_size[1], size)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(point_coords_trg, trg_size[0], trg_size[1], size)
        files.append(src_imname)
        files.append(trg_imname)
        
        
        
        vis = source_kps[:, 2] * target_kps[:, 2] > 0
        
        data_pair = {"src_imsize":[size, size],
                     "trg_imsize":[size, size],
                     "src_imname":src_imname,
                     "trg_imname":trg_imname,
                     "threshold":None,
                     "src_kps":(source_kps[vis].to(torch.int32)).numpy().tolist(),
                     "trg_kps":(target_kps[vis].to(torch.int32)).numpy().tolist()}
        
        data_category.append(data_pair)
    
    files = list(set(files))
    return files, data_category