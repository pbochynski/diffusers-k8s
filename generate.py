from argparse import ArgumentParser
from flux_schnell import FluxSchnell



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Generate images by prompt using Flux-schnell")
    parser.add_argument("prompt", 
                        type=str, 
                        nargs="+",
                        help="Prompt that be used during inference")
    parser.add_argument("--num_inference_steps",
                        type=int,
                        default=4,
                        help="Number of inference steps used during generating")
    parser.add_argument("--device", 
                        type=str,
                        default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="The device used during inference. Default: `None`")
    parser.add_argument("--enable_sequential_cpu_offload",
                        action="store_true",
                        help="Enables sequential cpu offload during inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    FluxSchnell(
        device=args.device,
        enable_sequential_cpu_offload=args.enable_sequential_cpu_offload
    ).generate(args.prompt,
               args.num_inference_steps)