python evaluations/fastvideodiffusion/scripts/eval.py \
    --calculate_fvd \
    --calculate_lpips \
    --calculate_psnr \
    --calculate_ssim \
    --eval_method "videogpt" \
    --eval_dataset "./evaluations/fastvideodiffusion/datasets/webvid_selected.csv" \
    --eval_video_dir "./evaluations/fastvideodiffusion/datasets/webvid" \
    --generated_video_dir "./evaluations/fastvideodiffusion/samples/opensora/sample_skip" \
    --device "0"