#!/usr/bin/env bash
#SBATCH --output=/Midgard/home/%u/.local/pytorch-CycleGAN-and-pix2pix/runs/%J_slurm_tv.out
#SBATCH --error=/Midgard/home/%u/.local/pytorch-CycleGAN-and-pix2pix/runs/%J_slurm_tv.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yipingx@kth.se
#SBATCH --constrain="khazadum"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10GB



nvidia-smi


RUNS_PATH="/Midgard/home/yipingx/.local/pytorch-CycleGAN-and-pix2pix/runs"
DATA_PATH="/local_storage/datasets/yipingx/prune_unfilled_norm_sep_256x256"


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"



. ~/miniconda3/etc/profile.d/conda.sh

conda activate pytorch-py3
python -m visdom.server --hostname=khazadum.csc.kth.se -port=9010 & 

python train_load_data.py --dataroot=/local_storage/datasets/yipingx/prune_unfilled_norm_sep_256x256 --name=sss2depth_256x256 --model=sss2depth --direction=AtoB \
--niter=500 --niter_decay=500 --batch_size=32 --preprocess=none --input_nc=1 --output_nc=1 --num_threads=8 --display_server=khazadum.csc.kth.se \
--display_port=9010 --max_dataset_size=100000 --display_id=100 --display_freq=100 --print_freq=100 --save_epoch_freq=50 --lambda_TV=0.0 --load_size=256 \
--no_dropout --no_flip --beta1=0.9 --ngf=32 --netG=resnet_7blocks