# Evaluation
## Metric
Calculate the following video quality metrics:

- **FVD**: [Frechét Video Distance](https://arxiv.org/pdf/1812.01717)
- **SSIM**: structural similarity index measure
- **LPIPS**: learned perceptual image patch similarity
- **PSNR**: peak-signal-to-noise ratio


## Dataset
### WebVid Dataset 🕸🎥
The [WebVid](https://github.com/m-bain/webvid) dataset is a large-scale text-video dataset containing 10 million video-text pairs scraped from stock footage sites. This dataset is used for large-scale pretraining to achieve state-of-the-art end-to-end retrieval. To download the WebVid dataset for evaluation, run the following command:

- Download webvid dataset for evaluation.
```shell
bash evaluations/fastvideodiffusion/eval_webvid.sh
```


## Latte

### generate videos
```shell
bash evaluations/fastvideodiffusion/scripts/latte/generate_eval_latte_dataset.sh
```
- Edit `eval` config in `evaluations/fastvideodiffusion/configs/latte/sample_skip.yaml`
  - The generated videos will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/latte/sample_skip.yaml`


### eval
```shell
bash evaluations/fastvideodiffusion/scripts/latte/eval_latte.sh
```
The evaluation results will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/latte/sample_skip.yaml`



## OpenSora

### generate videos
```shell
bash evaluations/fastvideodiffusion/scripts/opensora/generate_eval_opensora_dataset.sh
```
- Edit `eval` config in `evaluations/fastvideodiffusion/configs/opensora/sample_skip.yaml`
  - The generated videos will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/opensora/sample_skip.yaml`


### eval
```shell
bash evaluations/fastvideodiffusion/scripts/opensora/eval_opensora.sh
```
The evaluation results will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/opensora/sample_skip.yaml`




## Open-Sora-Plan

### generate videos
```shell
bash evaluations/fastvideodiffusion/scripts/opensora_plan/generate_eval_opensora_plan_dataset.sh
```
- Edit `eval` config in `evaluations/fastvideodiffusion/configs/opensora_plan/sample_65f_skip.yaml`
  - The generated videos will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/opensora_plan/sample_65f_skip.yaml`


### eval
```shell
bash evaluations/fastvideodiffusion/scripts/opensora_plan/eval_opensora_plan.sh
```
The evaluation results will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/opensora_plan/sample_65f_skip.yaml`

## Eval Config

The brief description of the key parameters in an example evaluation configuration file (`evaluations/fastvideodiffusion/configs/latte/sample_skip.yaml`):


- **eval**: True - Indicates that the evaluation mode is enabled.

- **eval_dataset**: ./evaluations/fastvideodiffusion/datasets/webvid_selected.csv - Specifies the path to the dataset used for evaluation.

- **save_img_path**: "./evaluations/fastvideodiffusion/samples/latte/sample_fvd/" - Specifies the directory where the generated videos and evaluation results will be saved.


# TODO
- eval code claim
- how to edit config
