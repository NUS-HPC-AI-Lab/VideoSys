# common_metrics_on_video_quality

You can easily calculate the following video quality metrics:

- **FVD**: Frechét Video Distance
- **SSIM**: structural similarity index measure
- **LPIPS**: learned perceptual image patch similarity
- **PSNR**: peak-signal-to-noise ratio

As for FVD
 1. The codebase refers to [MVCD](https://github.com/voletiv/mcvd-pytorch) and other websites and projects, I've just extracted the part of it that's relevant to the calculation. This code can be used to evaluate FVD scores for generative or predictive models. 
 2. Now **we have supported 2 pytorch-based FVD implementations** ([videogpt](https://github.com/wilson1yan/VideoGPT) and [styleganv](https://github.com/universome/stylegan-v), see issue [#4](https://github.com/JunyaoHu/common_metrics_on_video_quality/issues/4)). Their calculations are almost identical, and the difference is negligible.
 3. FVD calculates the feature distance between two sets of videos. (the I3D features of each video are do not go through the softmax() function, and the size of the last dimension is 400, not 1024)

And...

- This project supports grayscale and RGB videos.
- This project supports Ubuntu, but maybe something is wrong with Windows. If you can solve it, welcome any PR.
- **If the project cannot run correctly, please give me an issue or PR~**
- For more details see below Notice.

# Example

8 videos of a batch, 10 frames, 3 channels, 64x64 size.

```
import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 30
CHANNEL = 3
SIZE = 64
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")
device = torch.device("cpu")

import json
result = {}
result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
# result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt')
result['ssim'] = calculate_ssim(videos1, videos2)
result['psnr'] = calculate_psnr(videos1, videos2)
result['lpips'] = calculate_lpips(videos1, videos2, device)
print(json.dumps(result, indent=4))
```

It means we calculate:
    
- `FVD-frames[:10]`, `FVD-frames[:11]`, ..., `FVD-frames[:30]` 
- `avg-PSNR/SSIM/LPIPS-frame[0]`, `avg-PSNR/SSIM/LPIPS-frame[1]`, ..., `avg-PSNR/SSIM/LPIPS-frame[:30]`, and their std.

We cannot calculate `FVD-frames[:8]`, and it will pass when calculating, see ps.6.

The result shows: a all-zero matrix and a all-one matrix, their FVD-30 (FVD[:30]) is 151.17 (styleganv method). We also calculate their standard deviation. Other metrics are the same. And we use the calculation method of styleganv.

```
{
    "fvd": {
        "value": {
            "10": 570.07320378183,
            "11": 486.1906542471159,
            "12": 552.3373915075898,
            "13": 146.6242330185728,
            "14": 172.57268402948895,
            "15": 133.88932632116126,
            "16": 153.11023578170108,
            "17": 357.56400892781204,
            "18": 382.1335612721498,
            "19": 306.7100176942531,
            "20": 338.18221898178774,
            "21": 77.95587603163293,
            "22": 82.49997632357349,
            "23": 64.41624523513073,
            "24": 66.08097153313875,
            "25": 314.4341061962642,
            "26": 316.8616746151064,
            "27": 288.884418528541,
            "28": 287.8192683223724,
            "29": 152.15076552354864,
            "30": 151.16806952692093
        },
        "video_setting": [
            8,
            3,
            30,
            64,
            64
        ],
        "video_setting_name": "batch_size, channel, time, heigth, width"
    },
        "video_setting": [
            8,
            3,
            30,
            64,
            64
        ],
        "video_setting_name": "batch_size, channel, time, heigth, width"
    },
    "ssim": {
        "value": {
            "0": 9.999000099990664e-05,
            ...,
            "29": 9.999000099990664e-05
        },
        "value_std": {
            "0": 0.0,
            ...,
            "29": 0.0
        },
        "video_setting": [
            30,
            3,
            64,
            64
        ],
        "video_setting_name": "time, channel, heigth, width"
    },
    "psnr": {
        "value": {
            "0": 0.0,
            ...,
            "29": 0.0
        },
        "value_std": {
            "0": 0.0,
            ...,
            "29": 0.0
        },
        "video_setting": [
            30,
            3,
            64,
            64
        ],
        "video_setting_name": "time, channel, heigth, width"
    },
    "lpips": {
        "value": {
            "0": 0.8140146732330322,
            ...,
            "29": 0.8140146732330322
        },
        "value_std": {
            "0": 0.0,
            ...,
            "29": 0.0
        },
        "video_setting": [
            30,
            3,
            64,
            64
        ],
        "video_setting_name": "time, channel, heigth, width"
    }
}
```

# Notice

1. You should `pip install lpips` first.
3. Make sure the pixel value of videos should be in [0, 1].
2. If you have something wrong with downloading FVD pre-trained model, you should manually download any of the following and put it into FVD folder. 
    - `i3d_torchscript.pt` from [here](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt) 
    - `i3d_pretrained_400.pt` from [here](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI)
4. For grayscale videos, we multiply to 3 channels [as it says](https://github.com/richzhang/PerceptualSimilarity/issues/23#issuecomment-492368812).
5. We average SSIM when images have 3 channels, ssim is the only metric extremely sensitive to gray being compared to b/w.
6. Because the i3d model downsamples in the time dimension, `frames_num` should > 10 when calculating FVD, so FVD calculation begins from 10-th frame, like upper example.
7. You had better use `scipy==1.7.3/1.9.3`, if you use 1.11.3, **you will calculate a WRONG FVD VALUE!!!**
8. If you are running demo.py on a multi-GPU machine, remember to export CUDA_VISIBLE_DEVICES=0, see [here](https://github.com/JunyaoHu/common_metrics_on_video_quality/issues/13).

# Star Trend
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JunyaoHu/common_metrics_on_video_quality&type=Date)](https://star-history.com/#JunyaoHu/common_metrics_on_video_quality&Date)
