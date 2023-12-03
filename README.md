# How to Use

## Training

```console
$mkdir data
$poetry shell
$python diffusion_models/generate_training_data.py
$python diffusion_models/generate_training_data.py train
$python diffusion_models/score_based_model.py train
$python diffusion_models/diffusion_model.py train
```

## Sampling

```console
$python diffusion_models/diffusion_model.py sample --ckpt_path ./path_to_trained_model_weights.ckpt
```