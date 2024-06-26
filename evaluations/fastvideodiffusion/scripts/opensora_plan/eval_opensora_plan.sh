GPU_ID="0"

CUDA_VISIBLE_DEVICES=$GPU_ID python evaluations/fastvideodiffusion/scripts/eval.py \
    --calculate_lpips \
    --calculate_psnr \
    --calculate_ssim \
    --eval_method "videogpt" \
    --eval_video_dir "./evaluations/fastvideodiffusion/samples/opensora_plan/sample_65f" \
    --generated_video_dir "./evaluations/fastvideodiffusion/samples/opensora_plan/sample_65f_pab"
