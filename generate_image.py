import torch
import random
import argparse
import os
from diffusers import DiffusionPipeline

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate an image using a diffusion model.")
parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu", help="Device to run the model on (cpu, cuda, mps)")
parser.add_argument("--enable_sequential_cpu_offload", type=bool, default=True, help="Enable sequential CPU offload (true/false)")
parser.add_argument("--steps", type=int, default=1, help="Number of inference steps")
parser.add_argument("--prompt", type=str, default="A detailed macro photo of LEGO minifigures as software developers in an open-space office. The main figure in front holds a sign with \"Kyma\" written with blue font on white background. Minifigures are surrounded by desks, laptops, coffee mugs, and coding equipment. Realistic lighting highlights the vibrant colors and fine LEGO textures.", help="Prompt for the image generation")
parser.add_argument("--seed", type=int, help="Random seed for image generation")


args = parser.parse_args()

# Load model
model = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
device = torch.device(args.device)
if args.enable_sequential_cpu_offload:
    model.enable_sequential_cpu_offload(device=device)
model.enable_attention_slicing()

# Use provided seed or generate a random one
seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

# Generate image
prompt = args.prompt
image = model(
    prompt,
    num_inference_steps=args.steps,
    generator=torch.Generator("cpu").manual_seed(seed)
).images[0]

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Save image with seed in filename
output_filename = f"{output_dir}/img-{seed}-{args.steps}.png"
image.save(output_filename)

# Save prompt to a text file with the same filename
prompt_filename = f"{output_dir}/img-{seed}-{args.steps}.txt"
with open(prompt_filename, "w") as f:
    f.write(prompt)

print(f"Image saved to {output_filename}")
print(f"Prompt saved to {prompt_filename}")