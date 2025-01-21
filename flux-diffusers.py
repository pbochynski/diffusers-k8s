import torch
from diffusers import FluxPipeline
model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
device = torch.device("mps")
model.enable_sequential_cpu_offload(device=device) 
model.enable_attention_slicing()

prompt = "Hyper-realistic photography, rich and warm colors with deep reds, golds, and dark shadows. Heavy textures, intricate details, and ornate patterns evoke the luxury and grandeur of baroque art, a male group of soldiers wearing Traditional Japanese kimono with a wide obi belt in A train station in the middle of nowhere."
image = model(
    prompt,
    num_inference_steps=1,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("turtle-6.png")