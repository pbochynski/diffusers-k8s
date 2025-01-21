from argparse import ArgumentParser
import gradio as gr
from flux_schnell import FluxSchnell



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Generate image using Flux-schnell via Gradio")
    parser.add_argument("--enable_sequential_cpu_offload",
                        action="store_true",
                        help="Enables sequential cpu offload")
    parser.add_argument("--share",
                        action="store_true",
                        help="Allows Gradio to produce public link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    flux_schnell = FluxSchnell("mps",
                               False,
                               args.enable_sequential_cpu_offload)
    
    gr_interface = gr.Interface(
        fn=lambda prompt, num_inference_steps: flux_schnell.generate(prompt,
                                                                     num_inference_steps,
                                                                     False,
                                                                     False)[0],
        inputs=[
            gr.Textbox(lines=3,
                       placeholder="an image of a lion in Claude Monet style",
                       label="Prompt"),
            gr.Slider(minimum=1, maximum=50, step=1, value=4, label="Inference steps")
        ],
        outputs=gr.Image(type="pil", label="Generated image"),
        examples=[
            ["A painting titled 'Two young peasant women' in Camille Pissarro style",
             4,],
        ],
        title="Generate Image by Prompt using Flux-schnell",
        allow_flagging="never"
    )
    gr_interface.launch(share=args.share)

    