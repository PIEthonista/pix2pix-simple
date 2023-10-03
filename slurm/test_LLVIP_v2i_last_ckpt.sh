#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J test_TWCC_pix2pix_LLVIP_v2i_last_ckpt
#SBATCH -p gp4d
#SBATCH -e test_TWCC_pix2pix_LLVIP_v2i_last_ckpt.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 520550
python test.py --train_dir /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_Visible2Infrared_2__25_09_2023__003321 --model_weights_path /work/u5832291/yixian/pix2pix_simple/experiments/pix2pix_Visible2Infrared_2__25_09_2023__003321/weights/netG_model_epoch_100.pth --test_dataset_input_dir /work/u5832291/datasets/LLVIP/visible/test --test_dataset_target_dir /work/u5832291/datasets/LLVIP/infrared/test --inference_instance_folder_name test_LLVIP_v2i_last_ckpt_20231003
