from videosys import CogVideoConfig, VideoSysEngine
from videosys.models.cogvideo.pipeline import CogVideoPABConfig
import os

import gradio as gr
import numpy as np
import torch
from openai import OpenAI
from time import time

dtype = torch.bfloat16
sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = OpenAI()
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            model="glm-4-0520",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=250,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt

def load_model(enable_video_sys=False, pab_threshold=[100, 850], pab_gap=2):
    pab_config = CogVideoPABConfig(full_threshold=pab_threshold, full_gap=pab_gap)
    config = CogVideoConfig(world_size=1, enable_pab=enable_video_sys, pab_config=pab_config)
    engine = VideoSysEngine(config)
    return engine



def generate(engine, prompt, num_inference_steps=50, guidance_scale=6.0):
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")
    return f"./outputs/{prompt}.mp4"


with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               VideoSys Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               <a href="https://github.com/NUS-HPC-AI-Lab/VideoSys">🌐 Github</a> 
           </div>

           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only. 
            Users should strictly adhere to local laws and ethics.
            </div>
            <div style="text-align: center; font-size: 15px; font-weight: bold; color: magenta; margin-bottom: 20px;">
            💡 This demo only demonstrates single-device inference. To experience the full power of VideoSys, please deploy it with multiple devices.
            </div>
           """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", value="a bear hunting for prey", lines=5)
            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")

            with gr.Column():
                gr.Markdown(
                    "**Optional Parameters** (default values are recommended)<br>"
                    "Turn Inference Steps larger if you want more detailed video, but it will be slower.<br>"
                    "50 steps are recommended for most cases. will cause 120 seconds for inference.<br>"
                )
                with gr.Row():
                    num_inference_steps = gr.Number(label="Inference Steps", value=50)
                    guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                    pab_gap = gr.Number(label="PAB Gap", value=2, precision=0)
                    pab_threshold = gr.Textbox(label="PAB Threshold", value="100,850", lines=1)
                with gr.Row():
                    generate_button = gr.Button("🎬 Generate Video")
                    generate_button_vs = gr.Button("⚡️ Generate Video with VideoSys (Faster)")

        with gr.Column():
            with gr.Row():
                video_output = gr.Video(label="CogVideoX", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                elapsed_time = gr.Textbox(label="Elapsed Time", value="0s", visible=False)
            with gr.Row():
                video_output_vs = gr.Video(label="CogVideoX with VideoSys", width=720, height=480)
            with gr.Row():
                download_video_button_vs = gr.File(label="📥 Download Video", visible=False)
                elapsed_time_vs = gr.Textbox(label="Elapsed Time", value="0s", visible=False)

    def generate_vanilla(prompt, num_inference_steps, guidance_scale, progress=gr.Progress(track_tqdm=True)):
        # tensor = infer(prompt, num_inference_steps, guidance_scale, progress=progress)
        engine = load_model()
        t = time()
        video_path = generate(engine, prompt, num_inference_steps, guidance_scale)
        elapsed_time = time() - t
        video_update = gr.update(visible=True, value=video_path)
        elapsed_time = gr.update(visible=True, value=f"{elapsed_time:.2f}s")

        return video_path, video_update, elapsed_time

    def generate_vs(prompt, num_inference_steps, guidance_scale, threshold, gap, progress=gr.Progress(track_tqdm=True)):
        # tensor = infer(prompt, num_inference_steps, guidance_scale, progress=progress)
        threshold = [int(i) for i in threshold.split(",")]
        gap = int(gap)
        engine = load_model(enable_video_sys=True, pab_threshold=threshold, pab_gap=gap)
        t = time()
        video_path = generate(engine, prompt, num_inference_steps, guidance_scale)
        elapsed_time = time() - t
        video_update = gr.update(visible=True, value=video_path)
        elapsed_time = gr.update(visible=True, value=f"{elapsed_time:.2f}s")

        return video_path, video_update, elapsed_time


    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)

    generate_button.click(
        generate_vanilla,
        inputs=[prompt, num_inference_steps, guidance_scale],
        outputs=[video_output, download_video_button, elapsed_time],
    )

    generate_button_vs.click(
        generate_vs,
        inputs=[prompt, num_inference_steps, guidance_scale, pab_threshold, pab_gap],
        outputs=[video_output_vs, download_video_button_vs, elapsed_time_vs],
    )

    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7870, share=True)