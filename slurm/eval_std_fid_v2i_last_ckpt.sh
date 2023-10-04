#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J eval_std_fid
#SBATCH -p gp4d
#SBATCH -e test_eval_std_fid_v2i_last_ckpt.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 525799
srun python /work/u5832291/yixian/palette_scene2scene_rec/eval_std_fid.py -s /work/u5832291/datasets/LLVIP/infrared/test -d /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_Visible2Infrared_2__25_09_2023__003321/test_outputs/test_LLVIP_v2i_last_ckpt_20231003
# conda activate palette

# compute FID between two folders
# Found 3463 images in the folder /work/u5832291/datasets/LLVIP/infrared/test
# Found 3463 images in the folder /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_Visible2Infrared_2__25_09_2023__003321/test_outputs/test_LLVIP_v2i_last_ckpt_20231003
# make_dataset
#   Std FID: 335.82061693598257
#       FID: 335.9992119390168
# IS (mean): 1.8652983733467448
#  IS (std): 0.19109481213640075

