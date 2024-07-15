import spaces
import os
import torch
import random
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline

import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# Download the model files
ckpt_dir = snapshot_download(repo_id="John6666/pony-realism-v21main-sdxl")

# Load the models
vae = AutoencoderKL.from_pretrained(os.path.join(ckpt_dir, "vae"), torch_dtype=torch.float16)

pipe = StableDiffusionXLPipeline.from_pretrained(
    ckpt_dir,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe = pipe.to("cuda")

pipe.unet.set_attn_processor(AttnProcessor2_0())

# Define samplers
samplers = {
    "Euler a": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
    "DPM++ 2M": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True),
    "DPM++ SDE Karras": DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
}

DEFAULT_POSITIVE_PREFIX = "score_9, score_8_up, score_7_up, BREAK"
DEFAULT_POSITIVE_SUFFIX = "(masterpiece), best quality, very aesthetic, perfect face"
DEFAULT_NEGATIVE_PREFIX = "score_1, score_2, score_3, text"
DEFAULT_NEGATIVE_SUFFIX = "nsfw, (low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn"

# Initialize Florence model
device = "cuda" if torch.cuda.is_available() else "cpu"
florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to(device).eval()
florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)

# Prompt Enhancer
enhancer_medium = pipeline("summarization", model="gokaygokay/Lamini-Prompt-Enchance", device=device)
enhancer_long = pipeline("summarization", model="gokaygokay/Lamini-Prompt-Enchance-Long", device=device)

# Florence caption function
def florence_caption(image):
    # Convert image to PIL if it's not already
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    inputs = florence_processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text,
        task="<DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )
    return parsed_answer["<DETAILED_CAPTION>"]

# Prompt Enhancer function
def enhance_prompt(input_prompt, model_choice):
    if model_choice == "Medium":
        result = enhancer_medium("Enhance the description: " + input_prompt)
        enhanced_text = result[0]['summary_text']
    else:  # Long
        result = enhancer_long("Enhance the description: " + input_prompt)
        enhanced_text = result[0]['summary_text']
    
    return enhanced_text

@spaces.GPU(duration=120)
def generate_image(additional_positive_prompt, additional_negative_prompt, height, width, num_inference_steps,
                   guidance_scale, num_images_per_prompt, use_random_seed, seed, sampler, clip_skip, 
                   use_florence2, use_medium_enhancer, use_long_enhancer,
                   use_positive_prefix, use_positive_suffix, use_negative_prefix, use_negative_suffix,
                   input_image=None, progress=gr.Progress(track_tqdm=True)):
    
    if use_random_seed:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = int(seed)  # Ensure seed is an integer
    
    # Set the scheduler based on the selected sampler
    pipe.scheduler = samplers[sampler]
    
    # Set clip skip
    pipe.text_encoder.config.num_hidden_layers -= (clip_skip - 1)
    
    # Start with the default positive prompt prefix if enabled
    full_positive_prompt = DEFAULT_POSITIVE_PREFIX + ", " if use_positive_prefix else ""

    # Add Florence-2 caption if enabled and image is provided
    if use_florence2 and input_image is not None:
        florence2_caption = florence_caption(input_image)
        florence2_caption = florence2_caption.lower().replace('.', ',')
        additional_positive_prompt = f"{florence2_caption}, {additional_positive_prompt}" if additional_positive_prompt else florence2_caption

    # Enhance only the additional positive prompt if enhancers are enabled
    if additional_positive_prompt:
        enhanced_prompt = additional_positive_prompt
        if use_medium_enhancer:
            medium_enhanced = enhance_prompt(enhanced_prompt, "Medium")
            medium_enhanced = medium_enhanced.lower().replace('.', ',')
            enhanced_prompt = f"{enhanced_prompt}, {medium_enhanced}"
        if use_long_enhancer:
            long_enhanced = enhance_prompt(enhanced_prompt, "Long")
            long_enhanced = long_enhanced.lower().replace('.', ',')
            enhanced_prompt = f"{enhanced_prompt}, {long_enhanced}"
        full_positive_prompt += enhanced_prompt

    # Add the default positive suffix if enabled
    if use_positive_suffix:
        full_positive_prompt += f", {DEFAULT_POSITIVE_SUFFIX}"
    
    # Combine default negative prompt with additional negative prompt
    full_negative_prompt = ""
    if use_negative_prefix:
        full_negative_prompt += f"{DEFAULT_NEGATIVE_PREFIX}, "
    full_negative_prompt += additional_negative_prompt if additional_negative_prompt else ""
    if use_negative_suffix:
        full_negative_prompt += f", {DEFAULT_NEGATIVE_SUFFIX}"
    
    try:
        image = pipe(
            prompt=full_positive_prompt,
            negative_prompt=full_negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=torch.Generator(pipe.device).manual_seed(seed)
        ).images
        return image, seed, full_positive_prompt, full_negative_prompt
    except Exception as e:
        print(f"Error during image generation: {str(e)}")
        return None, seed, full_positive_prompt, full_negative_prompt

# Gradio interface
with gr.Blocks(theme='bethecloud/storj_theme') as demo:
    gr.HTML("""
    <h1 align="center">Pony Realism v21 SDXL - Text-to-Image Generation</h1>
    <p align="center">
    <a href="https://huggingface.co/John6666/pony-realism-v21main-sdxl/" target="_blank">[HF Model Page]</a>
    <a href="https://civitai.com/models/372465/pony-realism" target="_blank">[civitai Model Page]</a>
    <a href="https://huggingface.co/microsoft/Florence-2-base" target="_blank">[Florence-2 Model]</a>
    <a href="https://huggingface.co/gokaygokay/Lamini-Prompt-Enchance-Long" target="_blank">[Prompt Enhancer Long]</a>
    <a href="https://huggingface.co/gokaygokay/Lamini-Prompt-Enchance" target="_blank">[Prompt Enhancer Medium]</a>
    </p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            positive_prompt = gr.Textbox(label="Positive Prompt", placeholder="Add your positive prompt here")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Add your negative prompt here")
            
            with gr.Accordion("Advanced settings", open=False):
                height = gr.Slider(512, 2048, 1024, step=64, label="Height")
                width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                num_inference_steps = gr.Slider(20, 50, 30, step=1, label="Number of Inference Steps")
                guidance_scale = gr.Slider(1, 20, 6, step=0.1, label="Guidance Scale")
                num_images_per_prompt = gr.Slider(1, 4, 1, step=1, label="Number of images per prompt")
                use_random_seed = gr.Checkbox(label="Use Random Seed", value=True)
                seed = gr.Number(label="Seed", value=0, precision=0)
                sampler = gr.Dropdown(label="Sampler", choices=list(samplers.keys()), value="DPM++ SDE Karras")
                clip_skip = gr.Slider(1, 4, 2, step=1, label="Clip skip")
            
            with gr.Accordion("Captioner and Enhancers", open=False):
                input_image = gr.Image(label="Input Image for Florence-2 Captioner")
                use_florence2 = gr.Checkbox(label="Use Florence-2 Captioner", value=False)
                use_medium_enhancer = gr.Checkbox(label="Use Medium Prompt Enhancer", value=False)
                use_long_enhancer = gr.Checkbox(label="Use Long Prompt Enhancer", value=False)
            
            with gr.Accordion("Prefix and Suffix Settings", open=False):
                use_positive_prefix = gr.Checkbox(
                    label="Use Positive Prefix", 
                    value=True, 
                    info=f"Prefix: {DEFAULT_POSITIVE_PREFIX}"
                )
                use_positive_suffix = gr.Checkbox(
                    label="Use Positive Suffix", 
                    value=True, 
                    info=f"Suffix: {DEFAULT_POSITIVE_SUFFIX}"
                )
                use_negative_prefix = gr.Checkbox(
                    label="Use Negative Prefix", 
                    value=True, 
                    info=f"Prefix: {DEFAULT_NEGATIVE_PREFIX}"
                )
                use_negative_suffix = gr.Checkbox(
                    label="Use Negative Suffix", 
                    value=True, 
                    info=f"Suffix: {DEFAULT_NEGATIVE_SUFFIX}"
                )
            
            generate_btn = gr.Button("Generate Image")

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(label="Result", elem_id="gallery", show_label=False)
            seed_used = gr.Number(label="Seed Used")
            full_positive_prompt_used = gr.Textbox(label="Full Positive Prompt Used")
            full_negative_prompt_used = gr.Textbox(label="Full Negative Prompt Used")

    generate_btn.click(
        fn=generate_image,
        inputs=[
            positive_prompt, negative_prompt, height, width, num_inference_steps,
            guidance_scale, num_images_per_prompt, use_random_seed, seed, sampler,
            clip_skip, use_florence2, use_medium_enhancer, use_long_enhancer,
            use_positive_prefix, use_positive_suffix, use_negative_prefix, use_negative_suffix,
            input_image
        ],
        outputs=[output_gallery, seed_used, full_positive_prompt_used, full_negative_prompt_used]
    )

demo.launch(debug=True)