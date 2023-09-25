#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J TWCC_pix2pix_nerf_n2d
#SBATCH -p gp4d
#SBATCH -e train_TWCC_pix2pix_nerf_n2d.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 520550
python train.py --root_dir /work/u5832291/yixian/pix2pix_simple --project_name pix2pix_nerf_n2d_twcc --wandb_project palette_scene2scene --train_dataset_input_dir /work/u5832291/Palette-Image-to-Image-Diffusion-Models/datasets/nerf/train/night --train_dataset_target_dir /work/u5832291/Palette-Image-to-Image-Diffusion-Models/datasets/nerf/train/day --test_dataset_input_dir /work/u5832291/Palette-Image-to-Image-Diffusion-Models/datasets/nerf/test/night --test_dataset_target_dir /work/u5832291/Palette-Image-to-Image-Diffusion-Models/datasets/nerf/test/day --batch_size 2 --test_batch_size 1 --direction a2b --gpu_id 0 

