#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J eval_std_fid
#SBATCH -p gp4d
#SBATCH -e test_eval_std_fid_d2n_highest_psnr.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 525794
srun python /work/u5832291/yixian/palette_scene2scene_rec/eval_std_fid.py -s /work/u5832291/Palette-Image-to-Image-Diffusion-Models/datasets/nerf/test/night -d /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_nerf_d2n_twcc__26_09_2023__054952/test_outputs/test_nerf_d2n_highest_psnr_20231003
# conda activate palette

# compute FID between two folders
# Found 76 images in the folder /work/u5832291/Palette-Image-to-Image-Diffusion-Models/datasets/nerf/test/night
# Found 76 images in the folder /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_nerf_d2n_twcc__26_09_2023__054952/test_outputs/test_nerf_d2n_highest_psnr_20231003
# make_dataset
#   Std FID: 111.92679437492052
#       FID: 110.62262366513593
# IS (mean): 2.061370923412626
#  IS (std): 0.3028295497329185
