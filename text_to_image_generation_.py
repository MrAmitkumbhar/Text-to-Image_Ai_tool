import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# ✅ Replace with your Hugging Face token (create one at https://huggingface.co/settings/tokens)
authorization_token = "your_token_here"

model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # ✅ force CPU since you don't have CUDA

# Load model with float32 (needed for CPU)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    use_auth_token=authorization_token  # works for older versions; newer prefers huggingface-cli login
)
pipe.to(device)

# Prompt
prompt = input("Enter your prompt: ")

# Generate
image = pipe(prompt, guidance_scale=8.5).images[0]

# Show
plt.imshow(image)
plt.axis("off")
plt.show()
