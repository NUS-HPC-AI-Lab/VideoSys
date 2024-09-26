import os

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), ".tmp_outputs")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import uuid

import spaces

import gradio as gr
from videosys import CogVideoXConfig, CogVideoXPABConfig, VideoSysEngine


def load_model(model_name, enable_video_sys=False, pab_threshold=[100, 850], pab_range=2):
    pab_config = CogVideoXPABConfig(spatial_threshold=pab_threshold, spatial_range=pab_range)
    config = CogVideoXConfig(model_name, enable_pab=enable_video_sys, pab_config=pab_config)
    engine = VideoSysEngine(config)
    return engine


def generate(engine, prompt, num_inference_steps=50, guidance_scale=6.0):
    video = engine.generate(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).video[0]

    unique_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join("./.tmp_outputs", unique_filename)

    engine.save_video(video, output_path)
    return output_path


@spaces.GPU(duration=200)
def generate_vs(
    model_name,
    prompt,
    num_inference_steps,
    guidance_scale,
    threshold_start,
    threshold_end,
    gap,
    progress=gr.Progress(track_tqdm=True),
):
    threshold = [int(threshold_end), int(threshold_start)]
    gap = int(gap)
    engine = load_model(model_name, enable_video_sys=True, pab_threshold=threshold, pab_range=gap)
    video_path = generate(engine, prompt, num_inference_steps, guidance_scale)
    return video_path


css = """
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    margin: 0 auto;
    padding: 20px;
}


.container {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.column {
    flex: 1;
    min-width: 0;
}

.video-output {
    width: 100%;
    max-width: 720px;
    height: auto;
    margin: 0 auto;
}

.server-status {
    margin-top: 5px;
    padding: 5px;
    font-size: 0.8em;
}
.server-status h4 {
    margin: 0 0 3px 0;
    font-size: 0.9em;
}
.server-status .row {
    margin-bottom: 2px;
}
.server-status .textbox {
    min-height: unset !important;
}
.server-status .textbox input {
    padding: 1px 5px !important;
    height: 20px !important;
    font-size: 0.9em !important;
}
.server-status .textbox label {
    margin-bottom: 0 !important;
    font-size: 0.9em !important;
    line-height: 1.2 !important;
}
.server-status .textbox {
    gap: 0 !important;
}
.server-status .textbox input {
    margin-top: -2px !important;
}

@media (max-width: 768px) {
    .row {
        flex-direction: column;
    }
    .column {
        width: 100%;
    }
}
    .video-output {
        width: 100%;
        height: auto;
    }
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
        VideoSys for CogVideoX🤗
    </div>
    <div style="text-align: center; font-size: 15px;">
        🌐 Github: <a href="https://github.com/NUS-HPC-AI-Lab/VideoSys">https://github.com/NUS-HPC-AI-Lab/VideoSys</a><br>

        ⚠️ This demo is for academic research and experiential use only.
        Users should strictly adhere to local laws and ethics.<br>

        💡 This demo only demonstrates single-device inference. To experience the full power of VideoSys, please deploy it with multiple devices.<br><br>
        </div>
    </div>
    """
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", value="Sunset over the sea.", lines=2)

            with gr.Column():
                gr.Markdown("**Generation Parameters**<br>")
                with gr.Row():
                    model_name = gr.Radio(["THUDM/CogVideoX-2b"], label="Model Type", value="THUDM/CogVideoX-2b")
                with gr.Row():
                    num_inference_steps = gr.Slider(label="Inference Steps", maximum=50, value=50)
                    guidance_scale = gr.Slider(label="Guidance Scale", value=6.0, maximum=15.0)
                gr.Markdown("**Pyramid Attention Broadcast Parameters**<br>")
                with gr.Row():
                    pab_range = gr.Slider(
                        label="Broadcast Range",
                        value=2,
                        step=1,
                        minimum=1,
                        maximum=4,
                        info="Attention broadcast range.",
                    )
                    pab_threshold_start = gr.Slider(
                        label="Start Timestep",
                        minimum=500,
                        maximum=1000,
                        value=850,
                        step=1,
                        info="Broadcast start timestep (1000 is the fisrt).",
                    )
                    pab_threshold_end = gr.Slider(
                        label="End Timestep",
                        minimum=0,
                        maximum=500,
                        step=1,
                        value=100,
                        info="Broadcast end timestep (0 is the last).",
                    )
                with gr.Row():
                    generate_button_vs = gr.Button("⚡️ Generate Video with VideoSys")

        with gr.Column():
            with gr.Row():
                video_output_vs = gr.Video(label="CogVideoX with VideoSys", width=720, height=480)

    generate_button_vs.click(
        generate_vs,
        inputs=[
            model_name,
            prompt,
            num_inference_steps,
            guidance_scale,
            pab_threshold_start,
            pab_threshold_end,
            pab_range,
        ],
        outputs=[video_output_vs],
        concurrency_id="gen",
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=1)
    demo.launch()
