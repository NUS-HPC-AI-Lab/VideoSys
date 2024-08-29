import os

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), ".tmp_outputs")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import uuid

import GPUtil
import psutil
import torch

import gradio as gr
from videosys import CogVideoXConfig, CogVideoXPABConfig, VideoSysEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dtype = torch.float16


def load_model(enable_video_sys=False, pab_threshold=[100, 850], pab_range=2):
    pab_config = CogVideoXPABConfig(spatial_threshold=pab_threshold, spatial_range=pab_range)
    config = CogVideoXConfig(world_size=1, enable_pab=enable_video_sys, pab_config=pab_config)
    engine = VideoSysEngine(config)
    return engine


def generate(engine, prompt, num_inference_steps=50, guidance_scale=6.0):
    video = engine.generate(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).video[0]

    unique_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join("./.tmp_outputs", unique_filename)

    engine.save_video(video, output_path)
    return output_path


def get_server_status():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append(
            {
                "id": gpu.id,
                "name": gpu.name,
                "load": f"{gpu.load*100:.1f}%",
                "memory_used": f"{gpu.memoryUsed}MB",
                "memory_total": f"{gpu.memoryTotal}MB",
            }
        )

    return {"cpu": f"{cpu_percent}%", "memory": f"{memory.percent}%", "disk": f"{disk.percent}%", "gpu": gpu_info}


def generate_vanilla(prompt, num_inference_steps, guidance_scale, progress=gr.Progress(track_tqdm=True)):
    engine = load_model()
    video_path = generate(engine, prompt, num_inference_steps, guidance_scale)
    return video_path


def generate_vs(
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
    engine = load_model(enable_video_sys=True, pab_threshold=threshold, pab_range=gap)
    video_path = generate(engine, prompt, num_inference_steps, guidance_scale)
    return video_path


def get_server_status():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_memory = f"{gpu.memoryUsed}/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)"
        else:
            gpu_memory = "No GPU found"
    except:
        gpu_memory = "GPU information unavailable"

    return {
        "cpu": f"{cpu_percent}%",
        "memory": f"{memory.percent}%",
        "disk": f"{disk.percent}%",
        "gpu_memory": gpu_memory,
    }


def update_server_status():
    status = get_server_status()
    return (status["cpu"], status["memory"], status["disk"], status["gpu_memory"])


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
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", value="Sunset over the sea.", lines=4)

            with gr.Column():
                gr.Markdown("**Generation Parameters**<br>")
                with gr.Row():
                    num_inference_steps = gr.Number(label="Inference Steps", value=50)
                    guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                with gr.Row():
                    pab_range = gr.Number(
                        label="PAB Broadcast Range", value=2, precision=0, info="Broadcast timesteps range."
                    )
                    pab_threshold_start = gr.Number(label="PAB Start Timestep", value=850, info="Start from step 1000.")
                    pab_threshold_end = gr.Number(label="PAB End Timestep", value=100, info="End at step 0.")
                with gr.Row():
                    generate_button_vs = gr.Button("⚡️ Generate Video with VideoSys (Faster)")
                    generate_button = gr.Button("🎬 Generate Video (Original)")
                with gr.Column(elem_classes="server-status"):
                    gr.Markdown("#### Server Status")

                    with gr.Row():
                        cpu_status = gr.Textbox(label="CPU", scale=1)
                        memory_status = gr.Textbox(label="Memory", scale=1)

                    with gr.Row():
                        disk_status = gr.Textbox(label="Disk", scale=1)
                        gpu_status = gr.Textbox(label="GPU Memory", scale=1)

                    with gr.Row():
                        refresh_button = gr.Button("Refresh")

        with gr.Column():
            with gr.Row():
                video_output_vs = gr.Video(label="CogVideoX with VideoSys", width=720, height=480)
            with gr.Row():
                video_output = gr.Video(label="CogVideoX", width=720, height=480)

    generate_button.click(
        generate_vanilla,
        inputs=[prompt, num_inference_steps, guidance_scale],
        outputs=[video_output],
        concurrency_id="gen",
        concurrency_limit=1,
    )

    generate_button_vs.click(
        generate_vs,
        inputs=[prompt, num_inference_steps, guidance_scale, pab_threshold_start, pab_threshold_end, pab_range],
        outputs=[video_output_vs],
        concurrency_id="gen",
        concurrency_limit=1,
    )

    refresh_button.click(update_server_status, outputs=[cpu_status, memory_status, disk_status, gpu_status])
    demo.load(update_server_status, outputs=[cpu_status, memory_status, disk_status, gpu_status], every=1)

if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=1)
    demo.launch()
