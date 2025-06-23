import os
from typing import List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image, ImageOps
from safetensors import safe_open
from timm.models.crossvit import CrossAttention
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        return self.proj(image_embeds)


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, target_blocks=["block"],train_content_branch_only=False):#change
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.target_blocks = target_blocks
        self.train_content_branch_only = train_content_branch_only#change

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            print(name) # fix by wang for all names
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                selected = False
                print("current target_blocks:", self.target_blocks) # fix by wang for now target
                for block_name in self.target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        skip=False, #change
                        train_content_branch_only = self.train_content_branch_only  # change 传入冻结参数
                    ).to(self.device, dtype=torch.float16)
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        skip=True
                    ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    def get_multi_embeddings_no_detection(self, content_image, n_samples=3):
        """
        对整张 content_image 做 n_samples 次随机增广 (mirror, flip, etc) 并用 self.image_encoder 编码。
        生成 [n_samples, 1, hidden_dim] embedding 列表, 用于多主体注入。
        """
        emb_list = []
        for i in range(n_samples):
            # 1) 复制图像
            img_aug = content_image.copy()

            # 2) 简易随机变换 (镜像, 灰度)
            if random.random() < 0.5:
                img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:
                img_aug = ImageOps.grayscale(img_aug).convert("RGB")

            # changeprint 如果 debug 模式打开，则保存增广后的图像
            debug_filename = f"debug_content_aug_{i}.png"
            img_aug.save(debug_filename)
            print(f"[DEBUG] Saved augmented content image: {debug_filename}")

            # 3) 用 clip_image_processor 处理
            px = self.clip_image_processor(images=img_aug, return_tensors="pt").pixel_values
            px = px.to(self.device, dtype=torch.float16)

            # 4) 编码成 embedding
            with torch.inference_mode():
                vision_out = self.image_encoder(px).image_embeds  # shape [1, 1024] or [1,embedding_dim]

            emb_list.append(vision_out)  # 先保持 [1, hidden_dim], 之后在 generate() 里再 unsqueeze(1)

            # 打印调试信息，方便查看每个样本的统计数据
            print(f"[DEBUG] Sample {i}: embedding shape = {vision_out.shape}, mean = {vision_out.mean().item():.4f}, std = {vision_out.std().item():.4f}")
        return emb_list

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if content_prompt_embeds is not None:
            clip_image_embeds = clip_image_embeds - content_prompt_embeds

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            #if isinstance(attn_processor, IPAttnProcessor):
            if hasattr(attn_processor, "scale"):#change
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        neg_content_emb=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, content_prompt_embeds=neg_content_emb
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_image,
        content_image,#change
        multi_subject_emb,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        neg_content_emb=None,
        neg_content_prompt=None,
        neg_content_scale=1.0,
        **kwargs,
    ):
        # 保存风格参考图（pil_image）change
        pil_image.save("debug_style_input.png")
        print("[DEBUG] Saved debug_style_input.png")

        # 保存内容图（content_image）change
        content_image.save("debug_content_input.png")
        print("[DEBUG] Saved debug_content_input.png")

        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        if neg_content_emb is None:
            if neg_content_prompt is not None:
                with torch.inference_mode():
                    (
                        prompt_embeds_, # torch.Size([1, 77, 2048])
                        negative_prompt_embeds_,
                        pooled_prompt_embeds_, # torch.Size([1, 1280])
                        negative_pooled_prompt_embeds_,
                    ) = self.pipe.encode_prompt(
                        neg_content_prompt,
                        num_images_per_prompt=num_samples,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    pooled_prompt_embeds_ *= neg_content_scale
            else:
                pooled_prompt_embeds_ = neg_content_emb
        else:
            pooled_prompt_embeds_ = None

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image, content_prompt_embeds=pooled_prompt_embeds_)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        # 保存风格 embedding 对应的输入（转换前的风格图已保存）change
        print("[DEBUG] Obtained style image embeddings, shape:", image_prompt_embeds.shape)

        print("[DEBUG] Extracting content embeddings from content_image...")#change

        # 这里调用 get_multi_embeddings_no_detection, 参数 n_samples 由 multi_subject_emb 指定
        #multi_subject_emb = kwargs.get("multi_subject_emb", 4)
        content_emb_list = self.get_multi_embeddings_no_detection(content_image, n_samples=multi_subject_emb)
        # 将每个 embedding 从 [1, hidden_dim] 扩展为 [1,1,hidden_dim]，方便后续在注意力处理器中使用
        for i, emb in enumerate(content_emb_list):
            content_emb_list[i] = emb.unsqueeze(1)
            print(f"[DEBUG] Content embedding {i}: shape={content_emb_list[i].shape}, mean={emb.mean().item():.4f}")

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            print("[DEBUG] prompt_embeds shape:", prompt_embeds.shape)#changeprint
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        print("[DEBUG] Generator created with seed:", seed)#changeprint
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
# change IPAdapterComposer: 单一风格 + 多主体
class IPAdapterComposer(nn.Module):
    """
    演示:
      - style_image => style_emb (单一)
      - content_image => 多主体embedding (content_emb_list)
    在 generate() 里分别编码, 并通过 pipe(..., style_emb=..., content_emb_list=...) 传给 IPAttnProcessor2_0Composer
    """
    def __init__(self, pipe: StableDiffusionPipeline, device="cuda", num_tokens=4, cross_attention_dim=2048):
        super().__init__()
        self.pipe = pipe.to(device)
        self.device = device
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

        # 替换 U-Net cross-attn => IPAttnProcessor2_0Composer
        self.set_ip_adapter()

        # clip encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(
            self.device, dtype=torch.float16
        )
        self.clip_processor = CLIPImageProcessor()

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                hidden_size = unet.config.block_out_channels[0]

            if cross_dim is None:
                attn_procs[name] = AttnProcessor(hidden_size=hidden_size)
            else:
                # 使用 IPAttnProcessor2_0Composer
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    skip=False
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

    def encode_style_image(self, style_image: Image.Image):
        """
        对风格图只做一次编码 => [1,1024]
        """
        px = self.clip_processor(images=style_image, return_tensors="pt").pixel_values
        px = px.to(self.device, dtype=torch.float16)
        with torch.no_grad():
            emb = self.image_encoder(px).image_embeds
        return emb

    def encode_content_multi(self, content_image: Image.Image, n=3):
        """
        对原图(有多个主体)做 n 次增广 => n个embedding
        如果有检测/分割可替换
        """
        emb_list = []
        for i in range(n):
            aug_img = content_image.copy()
            # 简易增广
            if random.random() < 0.5:
                aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
            px = self.clip_processor(images=aug_img, return_tensors="pt").pixel_values
            px = px.to(self.device, dtype=torch.float16)
            with torch.no_grad():
                emb = self.image_encoder(px).image_embeds
            # shape [1,1024] => [1,1,1024]
            emb_list.append(emb.unsqueeze(1))
        return emb_list

    def set_scale(self, scale):
        """
        用于更新处理器的 scale
        """
        for attn_processor in self.pipe.unet.attn_processors.values():
            if hasattr(attn_processor, "scale"):
                attn_processor.scale = scale

    def generate(
        self,
        style_image: Image.Image,
        content_image: Image.Image,
        prompt="a man and a dog",
        negative_prompt="",
        scale=1.0,
        multi_subject_num=3,
        num_inference_steps=30,
        seed=42,
        **kwargs
    ):
        """
        1) style_image => style_emb (单一)
        2) content_image => 多主体 => content_emb_list
        3) 通过 pipe(..., style_emb=..., content_emb_list=...) => IPAttnProcessor2_0Composer
        """
        self.set_scale(scale)

        # 编码风格图 => style_emb
        style_emb = self.encode_style_image(style_image)  # [1,1024]
        print("[DEBUG] style_emb:", style_emb.shape)

        # 编码原图 => 多主体embedding
        content_emb_list = self.encode_content_multi(content_image, n=multi_subject_num)
        print("[DEBUG] content_emb_list len:", len(content_emb_list))

        # encode prompt
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=True
            )
        print("[DEBUG] prompt_embeds shape:", prompt_embeds.shape)

        # 直接传 prompt_embeds 给 pipeline
        generator = torch.Generator(device=self.device).manual_seed(seed)

        out = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=scale,
            style_emb=style_emb,             # 单一风格
            content_emb_list=content_emb_list,  # 多主体embedding
            generator=generator,
            **kwargs
        )
        return out.images