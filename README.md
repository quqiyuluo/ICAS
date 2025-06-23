# ICAS: IP-Adapter and ControlNet-based Attention Structure for Multi-Subject Style Transfer Optimization

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/document/YOUR_PAPER_ID) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) This repository contains the official implementation for the paper: **"ICAS: IP-Adapter and ControlNet-based Attention Structure for Multi-Subject Style Transfer Optimization"**.

Our work introduces ICAS, a novel framework for efficient and controllable multi-subject style transfer. It addresses the challenges of maintaining semantic fidelity for multiple subjects while applying a consistent style, without relying on computationally expensive inversion procedures or large-scale stylized datasets.

![Teaser Image](https://i.imgur.com/YOUR_TEASER_IMAGE.jpg) ## 核心特性 (Core Features)

- **Efficient Fine-tuning**: ICAS adaptively fine-tunes only the content injection branch of the pre-trained diffusion model, significantly enhancing the controllability of style while preserving identity semantics and avoiding the high cost of full model fine-tuning.
- **Decoupled Control over Structure and Style**: By combining IP-Adapter for adaptive style injection and ControlNet for structural conditionalization, our framework is able to faithfully preserve the global layout while precisely applying the style.
- **Cyclic Multi-Subject Embedding**: We propose a novel recurrent embedding mechanism that effectively solves the identity confusion and feature loss problems in multi-agent scenarios by iteratively injecting the features of each subject.
- **High-Quality Generation**: Extensive experiments and user studies demonstrate that ICAS outperforms existing state-of-the-art methods in terms of structure preservation, style consistency, and overall aesthetic quality.

## Setup

1.  **Clone this repository**:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
    cd YOUR_REPOSITORY
    ```

2.  **Install Dependencies**:
    Please make sure you have manually installed PyTorch from the [PyTorch official website](https://pytorch.org/get-started/locally/) according to your CUDA version. Then, install other dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download pre-trained model**:
    * Download our ICAS model weights: [Please provide your model weights download link here, such as Hugging Face Hub or Google Drive]
    * Download other necessary pre-trained models, such as SDXL base model, ControlNet and CLIP image encoder, and place them in the corresponding directories.

## Demo

You can use the provided `infer_style_controlnet.py` script to generate an example. Before running, please make sure you have prepared the content graph, style graph, and correctly configured the model path in the script.

```bash
python infer_style_controlnet.py \
    --content_image_path "path/to/your/content_image.jpg" \
    --style_image_path "path/to/your/style_image.png" \
    --output_path "results/output.png"
```
*We recommend that you modify the script parameters to be passed in through the command line for increased flexibility.*

## Evaluation

If you want to reproduce the quantitative evaluation results in our paper, you can use the `pyiqa` toolbox.

1.  **Prepare the dataset**: Prepare your test set as we describe in the paper and appendix.
2.  **Run the assessment**:
    ```bash
    # LPIPS
    pyiqa lpips -t /path/to/generated_images -r /path/to/original_content_images --device cuda
    
    # MANIQA
    pyiqa maniqa -t /path/to/generated_images --device cuda
    ```


## Disclaimer

This project strives to positively impact the domain of AI-driven image generation. Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and utilize it in a responsible manner. The developers do not assume any responsibility for potential misuse by users.

## Acknowledgements

Our work is based on or borrowed from several excellent open source projects, including [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [ControlNet](https://github.com/lllyasviel/ControlNet), and [Diffusers](https://github.com/huggingface/diffusers). We would like to express our sincere gratitude to the authors of these projects.
