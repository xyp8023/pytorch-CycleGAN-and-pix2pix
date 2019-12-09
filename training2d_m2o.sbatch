#!/usr/bin/env bash
#SBATCH --output=/Midgard/home/%u/.local/pytorch-CycleGAN-and-pix2pix/runs/%J_slurm_tv.out
#SBATCH --error=/Midgard/home/%u/.local/pytorch-CycleGAN-and-pix2pix/runs/%J_slurm_tv.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yipingx@kth.se
#SBATCH --constrain="khazadum|belegost"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10GB



nvidia-smi


RUNS_PATH="/Midgard/home/yipingx/.local/pytorch-CycleGAN-and-pix2pix/runs"
DATA_PATH="/local_storage/datasets/yipingx/prune_unfilled_norm_sep_29"


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"



. ~/miniconda3/etc/profile.d/conda.sh

conda activate pytorch-py3
python -m visdom.server --hostname=${SLURMD_NODENAME}.csc.kth.se -port=${PORT} & 

python train_sss2depth_m2o.py --dataroot=/local_storage/datasets/yipingx/prune_unfilled_norm_sep_29 --name=sss2depth_29to1_lr_${LR}_tv_${TV} --model=sss2depthm2o --direction=AtoB \
--niter=${NITER} --niter_decay=500 --batch_size=320 --preprocess=none --input_nc=1 --output_nc=1 --num_threads=8 --display_server=${SLURMD_NODENAME}.csc.kth.se \
--display_port=${PORT} --max_dataset_size=1000000 --display_id=1 --display_freq=100 --print_freq=100 --save_epoch_freq=50 --lambda_TV=${TV} --load_size=256 \
--no_dropout --no_flip --beta1=0.9 --ngf=32 --netG=resnet_7blocks_m2o --lr=${LR}
