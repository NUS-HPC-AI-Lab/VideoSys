<p align="center">
<img width="55%" alt="VideoSys" src="./assets/figures/logo.png?raw=true">
</p>
<h3 align="center">
An easy and efficient system for video generation
</h3>
<p align="center">| <a href="https://github.com/NUS-HPC-AI-Lab/VideoSys?tab=readme-ov-file#installation">Quick Start</a> | <a href="https://github.com/NUS-HPC-AI-Lab/VideoSys?tab=readme-ov-file#usage">Supported Models</a> | <a href="https://github.com/NUS-HPC-AI-Lab/VideoSys?tab=readme-ov-file#acceleration-techniques">Accelerations</a> | <a href="https://discord.gg/WhPmYm9FeG">Discord</a> | <a href="https://oahzxl.notion.site/VideoSys-News-42391db7e0a44f96a1f0c341450ae472?pvs=4">Media</a> | <a href="https://huggingface.co/VideoSys">HuggingFace Space</a> |
</p>

### Latest News 🔥
- [2024/09] Support [CogVideoX](https://github.com/THUDM/CogVideo), [Vchitect-2.0](https://github.com/Vchitect/Vchitect-2.0) and [Open-Sora-Plan v1.2.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan).
- [2024/08] 🔥 Evole from [OpenDiT](https://github.com/NUS-HPC-AI-Lab/VideoSys/tree/v1.0.0) to <b>VideoSys: An easy and efficient system for video generation</b>.
- [2024/08] 🔥 Release PAB paper: <b>[Real-Time Video Generation with Pyramid Attention Broadcast](https://arxiv.org/abs/2408.12588)</b>.
- [2024/06] 🔥 Propose Pyramid Attention Broadcast (PAB)[[paper](https://arxiv.org/abs/2408.12588)][[blog](https://oahzxl.github.io/PAB/)][[doc](./docs/pab.md)], the first approach to achieve <b>real-time</b> DiT-based video generation, delivering <b>negligible quality loss</b> without <b>requiring any training</b>.
- [2024/06] Support [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) and [Latte](https://github.com/Vchitect/Latte).
- [2024/03] 🔥 Propose Dynamic Sequence Parallel (DSP)[[paper](https://arxiv.org/abs/2403.10266)][[doc](./docs/dsp.md)], achieves **3x** speed for training and **2x** speed for inference in Open-Sora compared with sota sequence parallelism.
- [2024/03] Support [Open-Sora](https://github.com/hpcaitech/Open-Sora).
- [2024/02] 🎉 Release [OpenDiT](https://github.com/NUS-HPC-AI-Lab/VideoSys/tree/v1.0.0): An Easy, Fast and Memory-Efficent System for DiT Training and Inference.

# About

VideoSys is an open-source project that provides a user-friendly and high-performance infrastructure for video generation. This comprehensive toolkit will support the entire pipeline from training and inference to serving and compression.

We are committed to continually integrating cutting-edge open-source video models and techniques. Stay tuned for exciting enhancements and new features on the horizon!

## Installation

Prerequisites:

- Python >= 3.10
- PyTorch >= 1.13 (We recommend to use a >2.0 version)
- CUDA >= 11.6

We strongly recommend using Anaconda to create a new environment (Python >= 3.10) to run our examples:

```shell
conda create -n videosys python=3.10 -y
conda activate videosys
```

Install VideoSys:

```shell
git clone https://github.com/NUS-HPC-AI-Lab/VideoSys
cd VideoSys
pip install -e .
```


## Usage

VideoSys supports many diffusion models with our various acceleration techniques, enabling these models to run faster and consume less memory.

<b>You can find all available models and their supported acceleration techniques in the following table. Click `Code` to see how to use them.</b>

<table>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Train</th>
        <th rowspan="2">Infer</th>
        <th colspan="2">Acceleration Techniques</th>
        <th rowspan="2">Usage</th>
    </tr>
    <tr>
        <th><a href="https://github.com/NUS-HPC-AI-Lab/VideoSys?tab=readme-ov-file#dyanmic-sequence-parallelism-dsp-paperdoc">DSP</a></th>
        <th><a href="https://github.com/NUS-HPC-AI-Lab/VideoSys?tab=readme-ov-file#pyramid-attention-broadcast-pab-blogdoc">PAB</a></th>
    </tr>
    <tr>
        <td>Vchitect [<a href="https://github.com/Vchitect/Vchitect-2.0">source</a>]</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="./examples/vchitect/sample.py">Code</a></td>
    </tr>
    <tr>
        <td>CogVideoX [<a href="https://github.com/THUDM/CogVideo">source</a>]</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./examples/cogvideox/sample.py">Code</a></td>
    </tr>
    <tr>
        <td>Latte [<a href="https://github.com/Vchitect/Latte">source</a>]</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="./examples/latte/sample.py">Code</a></td>
    </tr>
    <tr>
        <td>Open-Sora-Plan [<a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan">source</a>]</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="./examples/open_sora_plan/sample.py">Code</a></td>
    </tr>
    <tr>
        <td>Open-Sora [<a href="https://github.com/hpcaitech/Open-Sora">source</a>]</td>
        <td align="center">🟡</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="./examples/open_sora/sample.py">Code</a></td>
    </tr>
</table>

You can also find easy demo with HuggingFace Space <a href="https://huggingface.co/VideoSys">[link]</a> and Gradio <a href="./gradio">[link]</a>.

## Acceleration Techniques

### Pyramid Attention Broadcast (PAB) [[paper](https://arxiv.org/abs/2408.12588)][[blog](https://arxiv.org/abs/2403.10266)][[doc](./docs/pab.md)]

Real-Time Video Generation with Pyramid Attention Broadcast

Authors: [Xuanlei Zhao](https://oahzxl.github.io/)<sup>1*</sup>,  [Xiaolong Jin]()<sup>2*</sup>,  [Kai Wang](https://kaiwang960112.github.io/)<sup>1*</sup>, and [Yang You](https://www.comp.nus.edu.sg/~youy/)<sup>1</sup> (* indicates equal contribution)

<sup>1</sup>National University of Singapore, <sup>2</sup>Purdue University

![method](./assets/figures/pab_method.png)

PAB is the first approach to achieve <b>real-time</b> DiT-based video generation, delivering <b>lossless quality</b> without <b>requiring any training</b>. By mitigating redundant attention computation, PAB achieves up to 21.6 FPS with 10.6x acceleration, without sacrificing quality across popular DiT-based video generation models including [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Latte](https://github.com/Vchitect/Latte) and [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan).

See its details [here](./docs/pab.md).

----

### Dyanmic Sequence Parallelism (DSP) [[paper](https://arxiv.org/abs/2403.10266)][[doc](./docs/dsp.md)]

![dsp_overview](./assets/figures/dsp_overview.png)

DSP is a novel, elegant and super efficient sequence parallelism for [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Latte](https://github.com/Vchitect/Latte) and other multi-dimensional transformer architecture.

It achieves **3x** speed for training and **2x** speed for inference in Open-Sora compared with sota sequence parallelism ([DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)). For a 10s (80 frames) of 512x512 video, the inference latency of Open-Sora is:

| Method | 1xH800 | 8xH800 (DS Ulysses) | 8xH800 (DSP) |
| ------ | ------ | ------ | ------ |
| Latency(s) | 106 | 45 | 22 |

See its details [here](./docs/dsp.md).


## Contributing

We welcome and value any contributions and collaborations. Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.

## Contributors

<a href="https://github.com/NUS-HPC-AI-Lab/VideoSys/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NUS-HPC-AI-Lab/VideoSys"/>
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NUS-HPC-AI-Lab/VideoSys&type=Date)](https://star-history.com/#NUS-HPC-AI-Lab/VideoSys&Date)

## Citation

```
@misc{videosys2024,
  author={VideoSys Team},
  title={VideoSys: An Easy and Efficient System for Video Generation},
  year={2024},
  publisher={GitHub},
  url = {https://github.com/NUS-HPC-AI-Lab/VideoSys},
}

@misc{zhao2024pab,
  title={Real-Time Video Generation with Pyramid Attention Broadcast},
  author={Xuanlei Zhao and Xiaolong Jin and Kai Wang and Yang You},
  year={2024},
  eprint={2408.12588},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2408.12588},
}

@misc{zhao2024dsp,
  title={DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers},
  author={Xuanlei Zhao and Shenggan Cheng and Chang Chen and Zangwei Zheng and Ziming Liu and Zheming Yang and Yang You},
  year={2024},
  eprint={2403.10266},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2403.10266},
}

@misc{zhao2024opendit,
  author={Xuanlei Zhao, Zhongkai Zhao, Ziming Liu, Haotian Zhou, Qianli Ma, and Yang You},
  title={OpenDiT: An Easy, Fast and Memory-Efficient System for DiT Training and Inference},
  year={2024},
  publisher={GitHub},
  url={https://github.com/NUS-HPC-AI-Lab/VideoSys/tree/v1.0.0},
}
```
