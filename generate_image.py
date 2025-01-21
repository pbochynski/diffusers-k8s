import torch
from diffusers import DiffusionPipeline

# Load model
model = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
device = torch.device("cuda")
model.enable_sequential_cpu_offload(device=device)
model.enable_attention_slicing()

# Generate image
prompt = "an image of a turtle in Picasso style"
image = model(
    prompt,
    num_inference_steps=1,
    generator=torch.Generator("cpu").manual_seed(1234)
).images[0]

# Save image
image.save("/output/turtle-3.png")
print("Image saved to /output/turtle-3.png")
