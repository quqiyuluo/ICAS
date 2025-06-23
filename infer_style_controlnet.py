import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

import cv2
from PIL import Image

from ip_adapter import IPAdapterXL

base_model_path = r"./stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = r"./sdxl_models/image_encoder"
ip_ckpt = r"./sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

controlnet_path = r"./diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])#change training

# style image
style_image = "./assets/style.png"
style_image = Image.open(style_image)
style_image.resize((512, 512))
content_image = Image.open("./assets/snow.jpeg").resize((512,512))
# control image
input_image = cv2.imread("./assets/snow.jpeg")
detected_map = cv2.Canny(input_image, 50, 200)
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

# generate image # saturation
images = ip_model.generate(pil_image=style_image,
                           content_image=content_image,
                           prompt="two persons, a man and a girl, masterpiece, best quality, high quality",
                           negative_prompt= "low contrast, text, watermark, lowres, low quality, worst quality, deformed, glitch, noisy, blurry",
                           scale=1.0,#1.0//
                           guidance_scale=3,#5
                           num_samples=1,
                           num_inference_steps=30,
                           seed=42,
                           image=canny_map,#Controlnet input
                           controlnet_conditioning_scale=0.8,
                           multi_subject_emb=2
                          )

images[0].save("result669_80.png")