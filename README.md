<p align="center">
<img width="55%" alt="VideoSys" src="./assets/figures/logo.png?raw=true">
</p>
<h3 align="center">
An easy and efficient system for video generation
</h3>
</p>

### Latest News 🔥
- [2024/08] 🔥<b>Release VideoSys: An easy and efficient system for video generation.</b>
- [2024/06] Propose Pyramid Attention Broadcast (PAB)[[paper](https://arxiv.org/abs/2408.12588)][[blog](https://oahzxl.github.io/PAB/)][[doc](./docs/pab.md)], the first approach to achieve <b>real-time</b> DiT-based video generation, delivering <b>negligible quality loss</b> without <b>requiring any training</b>.
- [2024/06] Support Open-Sora-Plan and Latte.
- [2024/03] Propose Dynamic Sequence Parallel (DSP)[[paper](https://arxiv.org/abs/2403.10266)][[doc](./docs/dsp.md)], achieves **3x** speed for training and **2x** speed for inference in Open-Sora compared with sota sequence parallelism.
- [2024/03] Support Open-Sora: Democratizing Efficient Video Production for All.
- [2024/02] Release OpenDiT: An Easy, Fast and Memory-Efficent System for DiT Training and Inference.

# About

VideoSys is an open-source project that provides a ease-of-use and high-performance infrastructure for video generation, including training, inference, serving and compression.

VideoSys will continue to integrate more open-source video models and techniques. Stay tuned for upcoming enhancements and additional features!

Authors: <a href="https://oahzxl.github.io/">Xuanlei Zhao</a>, <a href="https://kaiwang960112.github.io/">Kai Wang</a>, <a href="https://scholar.google.com/citations?user=wrhNmbIAAAAJ&hl">Geng Zhang</a>, <a href="https://viscent.dev">Yiqi Zhang</a>, <a href="https://shenggan.github.io">Shenggan Cheng</a>, <a href="https://www.cs.purdue.edu/people/graduate-students/jin509.html">Xiaolong Jin</a>, <a href="https://www.comp.nus.edu.sg/~youy/">Yang You</a>

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

<b>You can find all available models and their supported acceleration techniques in the following table. Click `Doc` to see how to use them.</b>

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
        <td>Open-Sora [<a href="https://github.com/hpcaitech/Open-Sora">source</a>]</td>
        <td align="center">🟡</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="./examples/open_sora/sample.py">Code</a></td>
    </tr>
    <tr>
        <td>Open-Sora-Plan [<a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan">source</a>]</td>
        <td align="center">❌</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="./examples/open_sora_plan/sample.py">Code</a></td>
    </tr>
    <tr>
        <td>Latte [<a href="https://github.com/Vchitect/Latte">source</a>]</td>
        <td align="center">❌</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="./examples/latte/sample.py">Code</a></td>
    </tr>
    <tr>
        <td>CogVideo [<a href="https://github.com/THUDM/CogVideo">source</a>]</td>
        <td align="center">❌</td>
        <td align="center">✅</td>
        <td align="center">❌</td>
        <td align="center">❌</td>
        <td align="center"><a href="./examples/cogvideo/sample.py">Code</a></td>
    </tr>
</table>

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


## Acknowledgement

Thanks [Zhongkai Zhao](https://www.linkedin.com/in/zhongkai-zhao-kk2000/), [Ziming Liu](https://maruyamaaya.github.io/), [Haotian Zhou](https://github.com/ht-zhou), [Qianli Ma](https://fazzie-key.cool/about/index.html) for their contributions. We also extend our gratitude to [Zangwei Zheng](https://zhengzangw.github.io/), [Fuzhao Xue](https://xuefuzhao.github.io/), [Shizun Wang](https://littlepure2333.github.io/home/), [Yuchao Gu](https://ycgu.site/), [Shenggui Li](https://franklee.xyz/), and [Haofan Wang](https://haofanwang.github.io/) for their invaluable advice.

## Contributing

If you encounter problems using VideoSys or have a feature request, feel free to create an issue! We also welcome pull requests from the community.

## Citation

```
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
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NUS-HPC-AI-Lab/VideoSys&type=Date)](https://star-history.com/#NUS-HPC-AI-Lab/VideoSys&Date)
