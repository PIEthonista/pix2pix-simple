#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J eval_std_fid
#SBATCH -p gp4d
#SBATCH -e test_eval_std_fid_i2v_highest_psnr.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 525795
srun python /work/u5832291/yixian/palette_scene2scene_rec/eval_std_fid.py -s /work/u5832291/datasets/LLVIP/visible/test -d /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_Infrared2Visible_2__25_09_2023__003330/test_outputs/test_LLVIP_i2v_highest_psnr_20231003
# conda activate palette

# compute FID between two folders
# Found 3463 images in the folder /work/u5832291/datasets/LLVIP/visible/test
# Found 3463 images in the folder /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_Infrared2Visible_2__25_09_2023__003330/test_outputs/test_LLVIP_i2v_highest_psnr_20231003
# make_dataset
#   Std FID: 331.86169188665883
#       FID: 334.3930172349417
# IS (mean): 1.7017579073232505
#  IS (std): 0.2779284057364851
