import base64
import requests
import json
import time

# Use the deployed endpoint URL
API_URL = "https://locationfind230--illusion-diffusion-inference-api-generate.modal.run"

# Read pattern image and encode as base64
with open("pattern_images/deneme.png", "rb") as f:
    b64_image = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image": b64_image,
    "prompt": "flowers, trees, nature",
    "negative_prompt": "low quality, blurry, distorted",
    "guidance_scale": 7.5,
    "controlnet_conditioning_scale": 1.2,
    "upscaler_strength": 1.0,
    "seed": 42,
    "sampler": "Euler"
}

print(f"Sending request to {API_URL}...")
start = time.time()
response = requests.post(API_URL, json=payload)
print(f"Response status: {response.status_code}")

if response.status_code == 200:
    with open("test_output2.png", "wb") as f:
        f.write(response.content)
    print(f"✅ Success! Image saved as test_output.png (Took {time.time() - start:.1f}s)")
else:
    print(f"❌ Error: {response.text}")
