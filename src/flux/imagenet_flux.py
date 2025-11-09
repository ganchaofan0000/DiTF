from diffusers import StableDiffusionPipeline, FluxPipeline
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
# from diffusers.models.unet_2d_condition import UNet2DConditionModel
from flux.model import Flux, FluxParams
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5
from flux.modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)
from diffusers import DDIMScheduler
from torch import Tensor, nn
import gc
import os
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import PILToTensor, ToPILImage
from einops import rearrange, repeat
import time

def prepare_txt(bs, t5, clip, prompt, device='cuda'):

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
    
    return txt.to(device), txt_ids.to(device), vec.to(device)

def prepare(img):
    bs, c, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    
    return img, img_ids.to(img.device)


class Featurizer:
    def __init__(self, name='flux-dev', null_prompt='', device='cuda'):
        
        t5 = load_t5(device, max_length=512)
        clip = load_clip(device)
        model = load_flow_model(name, device=device)
        ae = load_ae(name, device=device)

        self.t5 = t5
        self.clip = clip
        self.model = model
        self.ae = ae
        
        neg_prompt=""
        
        neg_prompt_embeds, neg_text_ids, neg_vec = prepare_txt(
                        bs=1,
                        t5=self.t5,
                        clip=self.clip,
                        prompt=neg_prompt)
        
        self.neg_prompt_embeds=neg_prompt_embeds
        self.neg_text_ids=neg_text_ids
        self.neg_vec=neg_vec
        
        

    # @torch.no_grad()
    # def forward(self,
    #             img_tensor,
    #             prompt='',
    #             t=261,
    #             up_ft_index=1,
    #             ensemble_size=8):
    #     '''
    #     Args:
    #         img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
    #         prompt: the prompt to use, a string
    #         t: the time step to use, should be an int in the range of [0, 1000]
    #         up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
    #         ensemble_size: the number of repeated images used in the batch to extract features
    #     Return:
    #         unet_ft: a torch tensor in the shape of [1, c, h, w]
    #     '''
    #     img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
    #     if prompt == self.null_prompt:
    #         prompt_embeds = self.null_prompt_embeds
    #     else:
    #         prompt_embeds = self.pipe._encode_prompt(
    #             prompt=prompt,
    #             device='cuda',
    #             num_images_per_prompt=1,
    #             do_classifier_free_guidance=False) # [1, 77, dim]
    #     prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
    #     unet_ft_all = self.pipe(
    #         img_tensor=img_tensor,
    #         t=t,
    #         up_ft_indices=[up_ft_index],
    #         prompt_embeds=prompt_embeds)
    #     unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
    #     unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
    #     return unet_ft


class Featurizer4Eval(Featurizer):
    def __init__(self, flux_id='flux-dev', null_prompt='', cat_list=['image'], ensemble_size=1):
        super().__init__(flux_id, null_prompt)
        
        ###FLUX
        with torch.no_grad():
            cat2prompt_embeds = {}
            for cat in cat_list:
                # prompt = f"a photo of a {cat}"
                prompt = f"a photo of a {cat}"
                # prompt = "The image captures a dynamic scene on a racetrack. A white sports car, possibly a vintage or classic model, is in motion, leaning into a turn with its front wheels angled sharply to the left. The car's bodywork is sleek, with a long hood and short rear deck, characteristic of high-performance vehicles designed for speed and agility."
                prompt_embeds, text_ids, vec = prepare_txt(
                    bs=ensemble_size,
                    t5=self.t5,
                    clip=self.clip,
                    prompt=prompt)
                cat2prompt_embeds[cat] = (prompt_embeds, text_ids, vec)
            self.cat2prompt_embeds = cat2prompt_embeds

        gc.collect()
        torch.cuda.empty_cache()


    @torch.no_grad()
    def forward(self,
                img,
                caption=None,
                category="image",
                img_size=[512, 512],
                t=261,
                ft_index=[28],
                ensemble_size=1,
                guidance=3.5,
                neg_catory="person"):
        
        
        # if img_size is not None:
        #     img = img.resize(img_size)
        img_tensor = (img / 255.0 - 0.5) * 2
        
        # img_tensor = torch.zeros_like(img_tensor)
        
        img_tensor = img_tensor.cuda() # ensem, c, h, w
        
        prompt_embeds, text_ids, vec = self.cat2prompt_embeds[category]
        
        ###
        device = img_tensor.device
        t=t/1000
        
        latents = self.ae.encode(img_tensor)
        
        latents = latents.to(torch.bfloat16)
        noise = torch.randn_like(latents).to(device)
        
        ### add noise
        latents_noisy = t * noise + (1.0 - t) * latents
        
        
        ensem, c, h, w = latents_noisy.shape
        
        img, img_ids = prepare(img=latents_noisy)
        
        t_vec = torch.full((img.shape[0],), t, dtype=img.dtype, device=img.device)
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        
        model_output = self.model.forward_feat(
                        img=img,
                        img_ids=img_ids,
                        txt=prompt_embeds,
                        txt_ids=text_ids,
                        y=vec,
                        timesteps=t_vec,
                        ft_indices=ft_index,
                        cat=category,
                        guidance=guidance_vec
                    )
        # end = time.time()

        # unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
        # unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
        # print(model_output)
        x_feat = model_output[0]
        x_feat = torch.mean(x_feat, dim=1)
        x_feat_ada = model_output[1]
        
        x_feat_ada = torch.mean(x_feat_ada, dim=1)
        
        
        return torch.cat((x_feat, x_feat_ada), dim=0)