from abc import abstractmethod

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


class VideoSysPipeline(DiffusionPipeline):
    def __init__(self):
        super().__init__()

    @staticmethod
    def set_eval_and_device(device: torch.device, *modules):
        for module in modules:
            module.eval()
            module.to(device)

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        """
        In diffusers, it is a convention to call the pipeline object.
        But in VideoSys, we will use the generate method for better prompt.
        This is a wrapper for the generate method to support the diffusers usage.
        """
        return self.generate(*args, **kwargs)
