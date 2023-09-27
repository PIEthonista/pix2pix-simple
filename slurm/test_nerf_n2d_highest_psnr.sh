#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_TWCC_pix2pix_nerf_n2d_highest_psnr
#SBATCH -p gp4d
#SBATCH -e test_TWCC_pix2pix_nerf_n2d_highest_psnr.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 520550
python test.py --train_dir /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_nerf_n2d_twcc__26_09_2023__054953 --model_weights_path /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_nerf_n2d_twcc__26_09_2023__054953/weights/netG_model_highest_psnr_8.812776786692803_epoch_100.pth --test_dataset_input_dir /work/u5832291/Palette-Image-to-Image-Diffusion-Models/datasets/nerf/test/night --test_dataset_target_dir /work/u5832291/Palette-Image-to-Image-Diffusion-Models/datasets/nerf/test/day
