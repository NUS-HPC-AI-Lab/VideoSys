# Evaluation of Pyramid Attention Broadcast

We first generate videos according to VBench's prompts.

And then calculate Vbench, PSNR, LPIPS and SSIM based on the video generated.

1. Generate video
```
cd eval/pab
python experiments/latte.py
python experiments/opensora.py
python experiments/open_sora_plan.py
```

2. Calculate Vbench score
```
# vbench is calculated independently
# get scores for all metrics
python vbench/run_vbench.py --video_path aaa --save_path bbb
# calculate final score
python vbench/cal_vbench.py --score_dir bbb
```

3. Calculate other metrics
```
# these metrics are calculated compared with original model
# gt video is the video of original model
# generated video is our methods's results
python common_metrics/eval.py --gt_video_dir aa --generated_video_dir bb
```
