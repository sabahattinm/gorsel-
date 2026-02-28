"""
IllusionDiffusion on Modal.com 🌀
Deploy the IllusionDiffusion HuggingFace Space as a serverless GPU API + Gradio Web UI.

Usage:
    modal run modal_app.py        # Download models
    modal deploy modal_app.py     # Deploy to Modal
"""

import io
import modal
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    image: str  # base64-encoded PNG/JPEG
    prompt: str
    negative_prompt: str = "low quality"
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 0.8
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    upscaler_strength: float = 1.0
    seed: int = -1
    sampler: str = "Euler"
    match_pattern_colors: bool = True
    convert_to_bw: bool = True

# ---------------------------------------------------------------------------
# Modal App & Image
# ---------------------------------------------------------------------------
app = modal.App("illusion-diffusion")

# Container image with all dependencies pre-installed
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.31.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "xformers==0.0.23.post1",
        "safetensors",
        "huggingface-hub==0.25.2",
        "scikit-image",
        "Pillow",
        "gradio~=4.44",
        "fastapi[standard]",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(
        "/Users/sabahattinmakine/gorsel/pattern_images",
        remote_path="/assets",
    )
)

# ---------------------------------------------------------------------------
# Model weight cache – download once into a Modal Volume
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("illusion-diffusion-models", create_if_missing=True)
MODEL_DIR = "/models"

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
VAE_MODEL = "stabilityai/sd-vae-ft-mse"
CONTROLNET_MODEL = "monster-labs/control_v1p_sd15_qrcode_monster"


def _download_models():
    """Download all model weights into the volume (runs once)."""
    import torch
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline

    print("⬇️  Downloading VAE...")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL, torch_dtype=torch.float16, cache_dir=MODEL_DIR
    )
    print("⬇️  Downloading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, torch_dtype=torch.float16, cache_dir=MODEL_DIR
    )
    print("⬇️  Downloading base model pipeline...")
    StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=torch.float16,
        cache_dir=MODEL_DIR,
    )
    print("✅  All models downloaded!")


@app.function(image=image, volumes={MODEL_DIR: volume}, timeout=900)
def download_models():
    """Download model weights into the volume."""
    _download_models()
    volume.commit()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size
    new_dimension = min(width, height)
    left = (width - new_dimension) / 2
    top = (height - new_dimension) / 2
    right = (width + new_dimension) / 2
    bottom = (height + new_dimension) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)
    return img


def _load_pipelines():
    """Load all pipelines to GPU. Returns (main_pipe, image_pipe, SAMPLER_MAP)."""
    import torch
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
    )

    SAMPLER_MAP = {
        "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(
            config, use_karras=True, algorithm_type="sde-dpmsolver++"
        ),
        "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    }

    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=torch.float16, cache_dir=MODEL_DIR)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=torch.float16, cache_dir=MODEL_DIR)
    main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=torch.float16,
        cache_dir=MODEL_DIR,
    ).to("cuda")
    main_pipe.enable_xformers_memory_efficient_attention()
    image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)
    image_pipe.enable_xformers_memory_efficient_attention()

    return main_pipe, image_pipe, SAMPLER_MAP


def _run_inference(main_pipe, image_pipe, SAMPLER_MAP, control_image, prompt,
                   negative_prompt="low quality", guidance_scale=7.5,
                   controlnet_conditioning_scale=0.8, control_guidance_start=0.0,
                   control_guidance_end=1.0, upscaler_strength=1.0, seed=-1,
                   sampler="Euler", match_pattern_colors=True,
                   convert_to_bw=True):
    """Core two-pass illusion diffusion pipeline."""
    import random
    import time
    import torch
    import numpy as np
    from skimage.exposure import match_histograms
    from PIL import Image

    start = time.time()
    
    # ----------------------------------------------------
    # Convert input to high-contrast Black&White if enabled
    # ----------------------------------------------------
    if convert_to_bw:
        print("🔲  Converting pattern to Black & White...")
        # L mode = grayscale. Converting back to RGB because the pipeline expects 3 channels.
        control_image = control_image.convert("L").convert("RGB")

    control_image_small = _center_crop_resize(control_image, (512, 512))
    control_image_large = _center_crop_resize(control_image, (1024, 1024))

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    my_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    generator = torch.Generator(device="cuda").manual_seed(my_seed)

    # Pass 1 — 512×512 latent generation
    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent",
    )

    # Pass 2 — upscale to 1024×1024 via img2img
    width = round(out["images"].shape[3] * 2)
    height = round(out["images"].shape[2] * 2)
    upscaled_latents = torch.nn.functional.interpolate(
        out["images"], size=(height, width), mode="nearest-exact"
    )

    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=20,
        strength=float(upscaler_strength),
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
    )

    result = out_image["images"][0]

    if match_pattern_colors:
        print("🎨  Applying color matching from pattern image...")
        # Match colors to the original control_image
        ref_image = np.array(control_image)
        gen_image = np.array(result)
        matched = match_histograms(gen_image, ref_image, channel_axis=-1)
        result = Image.fromarray(matched.astype(np.uint8))

    elapsed = time.time() - start
    print(f"⏱️  Generated in {elapsed:.1f}s (seed={my_seed})")
    return result, my_seed


# ---------------------------------------------------------------------------
# Inference Class — API endpoint
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="A10G",
    volumes={MODEL_DIR: volume},
    timeout=300,
    scaledown_window=120,
)
class Inference:
    """Serverless illusion artwork generator API."""

    @modal.enter()
    def setup(self):
        print("🔄  Loading models to GPU...")
        self.main_pipe, self.image_pipe, self.SAMPLER_MAP = _load_pipelines()
        print("✅  Models loaded!")

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str = "low quality",
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        upscaler_strength: float = 1.0,
        seed: int = -1,
        sampler: str = "Euler",
        match_pattern_colors: bool = True,
        convert_to_bw: bool = True,
    ) -> bytes:
        """Run the two-pass illusion diffusion pipeline. Returns PNG bytes."""
        from PIL import Image

        control_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result, my_seed = _run_inference(
            self.main_pipe, self.image_pipe, self.SAMPLER_MAP,
            control_image, prompt, negative_prompt, guidance_scale,
            controlnet_conditioning_scale, control_guidance_start,
            control_guidance_end, upscaler_strength, seed, sampler,
            match_pattern_colors, convert_to_bw
        )
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()

    @modal.fastapi_endpoint(method="POST")
    def api_generate(self, item: GenerateRequest):
        """
        POST JSON: {"image": "<base64>", "prompt": "...", ...}
        Returns PNG image.
        """
        import base64
        from fastapi.responses import Response
        from PIL import Image

        image_bytes = base64.b64decode(item.image)
        control_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result, _ = _run_inference(
            self.main_pipe, self.image_pipe, self.SAMPLER_MAP,
            control_image, item.prompt,
            item.negative_prompt,
            item.guidance_scale,
            item.controlnet_conditioning_scale,
            item.control_guidance_start,
            item.control_guidance_end,
            item.upscaler_strength,
            item.seed,
            item.sampler,
            item.match_pattern_colors,
            item.convert_to_bw,
        )

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Gradio Web UI
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_DIR: volume},
    timeout=600,
    scaledown_window=300,
)
@modal.asgi_app()
def web_ui():
    """Serve a Gradio interface for IllusionDiffusion."""
    # Patch gradio_client bug: "argument of type 'bool' is not iterable"
    # Triggered when a JSON schema has `additionalProperties: false` (bool).
    import gradio_client.utils as _gcu
    _orig_schema_fn = _gcu._json_schema_to_python_type
    def _safe_schema_fn(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_schema_fn(schema, defs)
    _gcu._json_schema_to_python_type = _safe_schema_fn

    import gradio as gr
    gr.set_static_paths(paths=["/assets"])

    print("🔄  Loading models for Gradio UI...")
    main_pipe, image_pipe, SAMPLER_MAP = _load_pipelines()
    print("✅  Models loaded for Gradio UI!")

    def inference(
        control_image, prompt, negative_prompt, guidance_scale,
        controlnet_conditioning_scale, control_guidance_start,
        control_guidance_end, upscaler_strength, seed, sampler,
        match_pattern_colors, convert_to_bw
    ):
        if control_image is None:
            raise gr.Error("Please select or upload an Input Illusion")
        if not prompt:
            raise gr.Error("Prompt is required")

        result, my_seed = _run_inference(
            main_pipe, image_pipe, SAMPLER_MAP,
            control_image, prompt, negative_prompt, guidance_scale,
            controlnet_conditioning_scale, control_guidance_start,
            control_guidance_end, upscaler_strength, seed, sampler,
            match_pattern_colors, convert_to_bw
        )
        return result, my_seed

    with gr.Blocks(title="IllusionDiffusion 🌀", theme=gr.themes.Soft(), max_file_size="100mb") as demo:
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1>IllusionDiffusion HQ 🌀</h1>
                <p style="font-size:16px;">Generate stunning high quality illusion artwork with Stable Diffusion</p>
                <p>Powered by <a href="https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster">Monster Labs QR Control Net</a> · Deployed on <a href="https://modal.com">Modal.com</a></p>
            </div>
            """
        )
        with gr.Row():
            with gr.Column():
                control_image = gr.Image(label="Input Illusion", type="pil")
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0, maximum=5.0, step=0.01, value=0.8,
                    label="Illusion strength", info="ControlNet conditioning scale",
                )
                gr.Examples(
                    examples=[
                        "/assets/checkers.png",
                        "/assets/checkers_mid.jpg",
                        "/assets/pattern.png",
                        "/assets/ultra_checkers.png",
                        "/assets/spiral.jpeg",
                        "/assets/funky.jpeg",
                    ],
                    inputs=control_image,
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Medieval village scene with busy streets and castle in the distance",
                )
                negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality")
                with gr.Accordion(label="Advanced Options", open=False):
                    guidance_scale = gr.Slider(0.0, 50.0, step=0.25, value=7.5, label="Guidance Scale")
                    sampler = gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="Euler", label="Sampler")
                    control_start = gr.Slider(0.0, 1.0, step=0.1, value=0, label="Start of ControlNet")
                    control_end = gr.Slider(0.0, 1.0, step=0.1, value=1, label="End of ControlNet")
                    strength = gr.Slider(0.0, 1.0, step=0.1, value=1, label="Upscaler Strength")
                    convert_to_bw = gr.Checkbox(value=True, label="Convert Template to B&W", info="Best illusions happen when the template is high-contrast black & white.")
                    match_pattern_colors = gr.Checkbox(value=True, label="Match Pattern Colors", info="Forces the output to inherit colors from the pattern image.")
                    seed = gr.Slider(-1, 9999999999, step=1, value=-1, label="Seed", info="-1 = random")
                run_btn = gr.Button("🎨 Generate", variant="primary")
            with gr.Column():
                result_image = gr.Image(label="Output", interactive=False)
                used_seed = gr.Number(label="Seed used", interactive=False)

        inputs = [
            control_image, prompt, negative_prompt, guidance_scale,
            controlnet_conditioning_scale, control_start, control_end,
            strength, seed, sampler, match_pattern_colors, convert_to_bw
        ]
        outputs = [result_image, used_seed]

        run_btn.click(inference, inputs=inputs, outputs=outputs)
        prompt.submit(inference, inputs=inputs, outputs=outputs)

    demo.queue(max_size=5)
    return demo.app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    """Download models to the volume."""
    print("📦  Downloading models to Modal Volume...")
    download_models.remote()
    print("✅  Models downloaded! Now deploy with: modal deploy modal_app.py")
