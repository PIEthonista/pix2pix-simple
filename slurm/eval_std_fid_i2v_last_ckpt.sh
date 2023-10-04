#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J eval_std_fid
#SBATCH -p gp4d
#SBATCH -e test_eval_std_fid_i2v_last_ckpt.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 525796
srun python /work/u5832291/yixian/palette_scene2scene_rec/eval_std_fid.py -s /work/u5832291/datasets/LLVIP/visible/test -d /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_Infrared2Visible_2__25_09_2023__003330/test_outputs/test_LLVIP_i2v_last_ckpt_20231003
# conda activate palette

# compute FID between two folders
# Found 3463 images in the folder /work/u5832291/datasets/LLVIP/visible/test
# Found 3463 images in the folder /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_Infrared2Visible_2__25_09_2023__003330/test_outputs/test_LLVIP_i2v_last_ckpt_20231003
# make_dataset
#   Std FID: 295.2359493004282
#       FID: 298.21437796186797
# IS (mean): 1.6469094742581962
#  IS (std): 0.26060188312218174

