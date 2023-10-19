#!\bin\bash

IMAGENET_DATASET_DIR='/path/to/imagenet/'
CARP400epMC='../../pretrained/carp/carp-400ep-mc/checkpoint0399.pth'

torchrun --nproc_per_node=4 eval_linear.py --pretrained_weights $CARP400epMC --data_path $IMAGENET_DATASET_DIR --batch_size_per_gpu 128 --gradient_accumulation_steps 4
