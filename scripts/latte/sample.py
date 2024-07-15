import opendit
from opendit import LatteConfig, LattePipeline

opendit.initialize(42)

config = LatteConfig()
pipeline = LattePipeline(config)

prompt = "Sunset over the sea."
video = pipeline(prompt).video[0]
pipeline.save_video(video, f"./outputs/{prompt[:50]}.mp4")
