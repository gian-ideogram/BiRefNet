#!/bin/bash
# Run script
# Settings of training & test for different tasks.
method="test" #"$1"
task=$(python3 config.py --print_task)
case "${task}" in
    'DIS5K') epochs=500 && val_last=50 && step=5 ;;
    'COD') epochs=150 && val_last=50 && step=5 ;;
    'HRSOD') epochs=150 && val_last=50 && step=5 ;;
    'Custom') epochs=$(( 244+101 )) && val_last=50 && step=10 ;; 
    'General-2K') epochs=250 && val_last=30 && step=2 ;;
    'Matting') epochs=150 && val_last=50 && step=5 ;;
esac

# Train
devices=$1

echo Training started at $(date)
# resume_weights_path='/home/gianfavero/projects/BiRefNet/ckpts/pre_trained_birefnets/BiRefNet-general-epoch_244.pth'
resume_weights_path="/home/gianfavero/projects/BiRefNet/ckpts/test/step_836.pth"

# Launch using accelerate
echo "Launching using accelerate..."
CUDA_VISIBLE_DEVICES=${devices}
accelerate launch --multi_gpu --gpu_ids ${devices} train_oom.py --ckpt_dir ckpts/${method} --epochs ${epochs} --resume ${resume_weights_path} --use_accelerate

echo Training finished at $(date)
