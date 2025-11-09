import os
from dataclasses import dataclass

import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from imwatermark import WatermarkEncoder
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder
import time
import cv2
import matplotlib.pyplot as plt
import gc
import copy
from sklearn.decomposition import PCA
from torchvision import transforms as T
from math import sqrt
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from einops import rearrange
from scipy.spatial.distance import cosine
from matplotlib.colors import ListedColormap

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path="/mnt/sda/chaofan/hg_models/FLUX.1-dev/flux1-dev.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path="/mnt/sda/chaofan/hg_models/FLUX.1-dev/ae.safetensors",
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("/mnt/nvme0n1/chaofan/hg_models/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("/mnt/sda/chaofan/hg_models/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]

        Returns:
            same as input but watermarked
        """
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
            image.device
        )
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image


def replace_center_feature_with_gpu(tensor, window_size, threshold=0.3):
    # 将输入张量转为 torch 并放置在指定设备 (GPU)
    # tensor = torch.tensor(tensor, device=device)
    tensor = tensor.squeeze(0).permute(1,2,0)
    # 获取张量尺寸
    height, width, channels = tensor.shape
    
    # 定义窗口半径
    radius = window_size // 2
    
    # 创建一个副本以避免修改原始张量
    modified_tensor = tensor.clone()
    
    # 手动填充张量的 height 和 width 维度（只在前两个维度进行对称填充）
    padded_tensor = torch.zeros((height + 2 * radius, width + 2 * radius, channels), device="cuda").to(torch.bfloat16)
    padded_tensor[radius:radius + height, radius:radius + width, :] = tensor
    
    # 将所有窗口的特征堆叠成一个 5D 张量 [batch, window_size, window_size, height, width]
    unfolded_windows = padded_tensor.unfold(0, window_size, 1).unfold(1, window_size, 1)
    
    # 将窗口展平为 [height, width, window_size*window_size, channels]
    # unfolded_windows = unfolded_windows.contiguous().view(height, width, -1, channels)
    
    unfolded_windows = rearrange(unfolded_windows, "h w c x y -> h w (x y) c")
    
    # 获取窗口的中心特征 [height, width, channels]
    center_indices = (window_size * window_size) // 2
    center_features = unfolded_windows[:, :, center_indices, :]  # 取出每个窗口的中心特征
    
    # 计算中心特征与窗口中其他所有特征的余弦相似度
    # 使用广播机制计算相似度 [height, width, window_size*window_size-1]
    similarities = F.cosine_similarity(
        center_features.unsqueeze(2),  # [height, width, 1, channels]
        unfolded_windows,  # [height, width, window_size*window_size, channels]
        dim=-1
    )
    
    # 去除中心特征本身的相似度
    similarities = torch.cat((similarities[:, :, :center_indices], similarities[:, :, center_indices+1:]), dim=2)
    
    # 计算平均相似度 [height, width]
    avg_similarities = similarities.mean(dim=-1)
    
    # 创建一个 mask，标记那些需要替换的中心特征 (True 表示需要替换)
    mask = avg_similarities < threshold
    
    # 计算每个窗口内的非中心特征的均值 [height, width, channels]
    non_center_features = torch.cat((unfolded_windows[:, :, :center_indices], unfolded_windows[:, :, center_indices+1:]), dim=2)
    mean_features = non_center_features.mean(dim=2)
    
    # 用均值替换中心特征
    # modified_tensor[mask] = mean_features[mask]
    inner_mask = mask[radius:-radius, radius:-radius]
    modified_tensor[radius:-radius, radius:-radius][inner_mask] = mean_features[radius:-radius, radius:-radius][inner_mask]
    
    modified_tensor = modified_tensor.permute(2,0,1).unsqueeze(0)
    
    return modified_tensor  # 返回到 CPU 并转换为 numpy 数组

def calculate_similarity(tensor, window_size, save_dir, img_name):
    
    
    # 获取特征张量的尺寸
    
    tensor = tensor.float().squeeze(0).permute(1,2,0).cpu().numpy()
    height, width, channels = tensor.shape
    
    # 定义n x n窗口的半径
    radius = window_size // 2
    
    # 初始化结果矩阵，用于存储每个位置的平均相似度
    similarity_map = np.ones((height, width))
    
    # 遍历每个位置作为中心点
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            # 取出n x n窗口
            window = tensor[i - radius:i + radius + 1, j - radius:j + radius + 1, :]
            
            # 获取中心特征
            center_feature = tensor[i, j, :]
            
            # 计算窗口中每个特征与中心特征的余弦相似度
            similarities = []
            for m in range(window_size):
                for n in range(window_size):
                    if m == radius and n == radius:
                        continue  # 跳过中心点本身
                    feature = window[m, n, :]
                    sim = 1 - cosine(center_feature, feature)  # 1 - cosine 距离即为余弦相似度
                    similarities.append(sim)
            
            # 计算平均相似度
            avg_similarity = np.mean(similarities)
            similarity_map[i, j] = avg_similarity
            
    visual_hotmap(similarity_map, save_dir=save_dir, img_name=img_name)
    
    # return similarity_map

def visual_hotmap(feature_map, save_dir=None, img_name=None):

    # 生成随机的特征图
    # feature_map = np.random.rand(32, 32, 1280)
    # feature_map = feature_map.squeeze(0).permute(1,2,0).cpu().numpy()
    # # print(feature_map.shape)
    # # 计算L2范数
    # l2_norm = np.linalg.norm(feature_map, axis=2)
    
    # print(np.max(l2_norm), np.min(l2_norm))

    # 可视化L2范数
    plt.imshow(feature_map, cmap='hot')
    plt.colorbar()
    plt.title('L2 Norm Visualization')
    plt.xlabel('Width')
    plt.ylabel('Height')
    if save_dir is not None:
        if not os.path.exists(save_dir):
            # 如果目录不存在，则创建它
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "%s.jpg"%img_name)
    
        # 保存为图片
        plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭图形以释放内存


def visual_matching(A, B, img1_path, img2_path, cat):
    # 转换颜色通道，从BGR到RGB
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1_name = (img1_path.split('/')[-1]).split('.')[0]
    img2_name = (img2_path.split('/')[-1]).split('.')[0]
    
    A = A[:10]
    B = B[:10]
    
    
    save_dir = "./matching_visualization/flux_denoise/image_pairs/%s/"%cat
    
    # 判断目录是否存在
    if not os.path.exists(save_dir):
        # 如果目录不存在，则创建它
        os.makedirs(save_dir)
        
    
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 不同的颜色
    # colors = plt.cm.get_cmap('hsv', len(A))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 在图像上绘制点
    ax1.imshow(img1_rgb)
    ax2.imshow(img2_rgb)

    for i, ((xa, ya), (xb, yb)) in enumerate(zip(A, B)):
        color = colors[i]
        ax1.scatter(xa, ya, color=color, label=f'Point {i+1}' if i == 0 else "")
        ax2.scatter(xb, yb, color=color, label=f'Point {i+1}' if i == 0 else "")

    # 保存图像
    plt.savefig(os.path.join(save_dir, "%s_%s.jpg"%(img1_name, img2_name)))

def visualize_and_save_features_pca_pair(src_ft, src_ft_in, trg_ft, img1_path, img2_path, cat):
    
    img1_name = (img1_path.split('/')[-1]).split('.')[0]
    img2_name = (img2_path.split('/')[-1]).split('.')[0]
    save_dir = "./matching_visualization/flux_pad_in/feat_pairs/%s/"%cat
    
    
    calculate_similarity(src_ft_in, window_size=3, save_dir="./matching_visualization/flux_pad_in/hotmap_pairs_w3/%s/"%cat, img_name="%s_%s"%(img1_name, img2_name))
    
    # b,c,h,w = src_ft.shape
    # # visual_hotmap(src_ft, save_dir=save_dir, img_name=img1_name)
    # # print(src_ft.shape)
    # # src_ft = nn.Upsample(size=(2*h, 2*w), mode='bilinear')(src_ft)
    
    # src_ft_copy=src_ft.clone()
    
    # src_ft = (src_ft).permute(0,2,3,1)
    # # src_ft =  F.normalize(src_ft, dim=-1)
    # src_ft = src_ft.reshape((b, -1, src_ft.shape[-1]))[0]
    
    # # trg_ft = nn.Upsample(size=(2*h, 2*w), mode='bilinear')(trg_ft)
    
    # trg_ft = (trg_ft).permute(0,2,3,1)
    # # trg_ft =  F.normalize(trg_ft, dim=-1)
    # trg_ft = trg_ft.reshape((b, -1, trg_ft.shape[-1]))[0]
    # num_tokens=src_ft.shape[0]
    # feature_maps=torch.cat((src_ft, trg_ft), dim=0)
    
    # # img1_name = (img1_path.split('/')[-1]).split('.')[0]
    # # img2_name = (img2_path.split('/')[-1]).split('.')[0]
    
    # # visual_hotmap(src_ft_copy, save_dir="./matching_visualization/flux_cfg_delta/hotmap/%s/"%cat, img_name=img1_name)
    
    # # calculate_similarity(src_ft_copy, window_size=3, save_dir="./matching_visualization/flux_1014/hotmap/%s/"%cat, img_name=img1_name)
    # # 判断目录是否存在
    # if not os.path.exists(save_dir):
    #     # 如果目录不存在，则创建它
    #     os.makedirs(save_dir)
    
    # feature_maps_fit_data = feature_maps
    # feature_maps_transform_data = copy.deepcopy(feature_maps)
    
    # feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    # pca = PCA(n_components=3)
    # pca.fit(feature_maps_fit_data)
    # feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    # pca_img_list=[]
    # for j in [0,1]:
    #     feature_maps_pca1 = feature_maps_pca[num_tokens*j:num_tokens*(j+1)]
    #     pca_img = feature_maps_pca1  # (H * W) x 3
    #     # print(feature_maps_pca1.shape)
    #     h = w = int(sqrt(pca_img.shape[0]))
    #     pca_img = pca_img.reshape(h, w, 3)
    #     pca_img_min = pca_img.min(axis=(0, 1))
    #     pca_img_max = pca_img.max(axis=(0, 1))
    #     pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
    #     pca_img_list.append(pca_img)
    # pca_img = np.concatenate(pca_img_list, axis=1)
    # pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    # pca_img = T.Resize((512, 1024), interpolation=T.InterpolationMode.NEAREST)(pca_img)
    # pca_img.save(os.path.join(save_dir, "%s_%s.jpg"%(img1_name, img2_name)))

def resize(img, target_res=224, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
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
    # kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale


def preprocess_data(data, size=512):
    
    source_size = data['src_imsize'][:2]
    target_size = data['trg_imsize'][:2]
    
    src_kps = torch.tensor(data['src_kps']).float()
    trg_kps = torch.tensor(data['trg_kps']).float()
    
    source_kps, src_x, src_y, src_scale = preprocess_kps_pad(src_kps, source_size[0], source_size[1], size)
    target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(trg_kps, target_size[0], target_size[1], size)
    
    trg_bndbox = np.asarray(data['trg_bndbox'])
    trg_bndbox = trg_bndbox * trg_scale
    
    data['src_imsize'] = [size, size]
    data['trg_imsize'] = [size, size]
    data['trg_bndbox'] = trg_bndbox
    data['src_kps'] = (source_kps.to(torch.int32)).numpy().tolist()
    data['trg_kps'] = (target_kps.to(torch.int32)).numpy().tolist()
    
    return data

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

def visual_matching_pad(args, A, B, img1_path, img2_path, cat):
    # 转换颜色通道，从BGR到RGB
    img1_name = (img1_path.split('/')[-1]).split('.')[0]
    img2_name = (img2_path.split('/')[-1]).split('.')[0]
    
    
    save_dir = "./matching_visualization/flux_pad/image_pairs/%s/"%cat

    # 判断目录是否存在
    if not os.path.exists(save_dir):
        # 如果目录不存在，则创建它
        os.makedirs(save_dir)
    
    src_img = Image.open(img1_path).convert('RGB')
    src_img = resize(src_img, args.img_size[0], resize=True, to_pil=True)

    trg_img = Image.open(img2_path).convert('RGB')
    trg_img = resize(trg_img, args.img_size[0], resize=True, to_pil=True)

    fig = draw_correspondences_gathered(A, B, src_img, trg_img)

    # 保存图像
    fig.savefig(os.path.join(save_dir, "%s_%s.jpg"%(img1_name, img2_name)))
    plt.close(fig)

# A fixed 48-bit message that was chosen at random
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)

from pathlib import Path
from torchvision import transforms

def load_video(video_folder: str, resize=None, num_frames=None):
    """
    Loads video from folder, resizes frames as desired, and outputs video tensor.

    Args:
        video_folder (str): folder containing all frames of video, ordered by frames index.
        resize (tuple): desired H' x W' dimensions. Defaults to (432, 768).
        num_frames (int): number of frames in video. Defaults to None.

    Returns:
        return_dict: Dictionary video tensor of shape: T x 3 x H' x W'.
    """
    path = Path(video_folder)
    input_files = sorted(list(path.glob("*.jpg")) + list(path.glob("*.png")))
    input_files = input_files[:num_frames] if num_frames is not None else input_files

    resh, resw = resize if resize is not None else (None, None)
    video = []
    
    for file in input_files:
        if resize is not None:
            video.append(transforms.ToTensor()(Image.open(str(file)).resize((resw, resh), Image.LANCZOS)))
        else:
            video.append(transforms.ToTensor()(Image.open(str(file))))
    
    return torch.stack(video)

def add_config_paths(data_path, config):
    # preprocessing
    config['video_folder'] = os.path.join(data_path, "video")
    config['trajectories_file'] = os.path.join(data_path, "of_trajectories", "trajectories.pt")
    config['unfiltered_trajectories_file'] = os.path.join(data_path, "of_trajectories", "trajectories_wo_direct_filter.pt")
    config['fg_trajectories_file'] = os.path.join(data_path, "of_trajectories", "fg_trajectories.pt")
    config['bg_trajectories_file'] = os.path.join(data_path, "of_trajectories", "bg_trajectories.pt")
    
    config['embed_video_path'] = os.path.join(data_path, "flux_embeddings", "flux_embed_video.pt")
    config['bb_dir'] = os.path.join(data_path, "flux_best_buddies")
    
    # model
    # outpts
    config['trajectories_dir'] = os.path.join(data_path, "trajectories_flux")
    config['occlusions_dir'] = os.path.join(data_path, "occlusions_flux")
    # config['trajectories_dir'] = os.path.join(data_path, "trajectories_raw")
    # config['occlusions_dir'] = os.path.join(data_path, "occlusions_raw")
    config['grid_trajectories_dir'] = os.path.join(data_path, "grid_trajectories")
    config['grid_occlusions_dir'] = os.path.join(data_path, "grid_occlusions")
    config['model_vis_dir'] = os.path.join(data_path, "visualizations")
    return config