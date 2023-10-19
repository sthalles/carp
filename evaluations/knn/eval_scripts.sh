#!\bin\bash

DATASET_DIR='/path/to/datasets/'

CARP200ep='../../pretrained/carp/carp-200ep/checkpoint0199.pth'
CARP400ep='../../pretrained/carp/carp-400ep/checkpoint0399.pth'
CARP200epMC='../../pretrained/carp/carp-200ep-mc/checkpoint0199.pth'
CARP400epMC='../../pretrained/carp/carp-400ep-mc/checkpoint0399.pth'

# # kNN evaluation for other SSL methods
# torchrun --nproc_per_node=4 eval_knn_flowers102.py --model obow --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/obow/tochvision_resnet50_student_K8192_epoch200.pth.tar
# torchrun --nproc_per_node=4 eval_knn_aircraft.py --model selav2 --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/sela_v2/selav2_400ep_2x224_pretrain.pth.tar
# torchrun --nproc_per_node=4 eval_knn_cars.py --model infomin --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/infomin/InfoMin_800.pth
# torchrun --nproc_per_node=4 eval_knn_stl10.py --model swav --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/swav/swav_800ep_pretrain.pth.tar
# torchrun --nproc_per_node=4 eval_knn_oxford_pet.py --model dino --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/dino/dino_resnet50_pretrain_full_checkpoint.pth
# torchrun --nproc_per_node=4 eval_knn_stl10.py --model deepclusterv2 --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/deepcluster_v2/deepclusterv2_800ep_pretrain.pth.tar
# torchrun --nproc_per_node=4 eval_knn_GTSRB.py --model triplet --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/triplet/release_ep940.pth
# torchrun --nproc_per_node=4 eval_knn_food101.py --model mocov3 --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/moco_v3/r-50-1000ep.pth.tar 
# torchrun --nproc_per_node=4 eval_knn_stl10.py --model barlowtwins --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../pretrained/barlowtwins/resnet50.pth
# torchrun --nproc_per_node=4 eval_knn_stl10.py --model carp --arch resnet50 --data_path $DATASET_DIR --pretrained_weights ../../methods/CARP_DINO/runs/Apr05_12-37-35_gpu-8/checkpoint0199.pth

# # CARP kNN evaluation
# torchrun --nproc_per_node=4 eval_knn_imagenet.py --model carp --arch resnet50 --data_path $DATASET_DIR --pretrained_weights $CARP200ep
# torchrun --nproc_per_node=4 eval_knn_imagenet.py --model carp --arch resnet50 --data_path $DATASET_DIR --pretrained_weights $CARP400ep
# torchrun --nproc_per_node=4 eval_knn_imagenet.py --model carp --arch resnet50 --data_path $DATASET_DIR --pretrained_weights $CARP200epMC
# torchrun --nproc_per_node=4 eval_knn_imagenet.py --model carp --arch resnet50 --data_path $DATASET_DIR --pretrained_weights $CARP400epMC

# #### Image retrieval and copy detection evaluation ####
# COPYDAYS_DATASET_DIR='/path/to/datasets/copydays'
# OXFORD_PARIS_DATASET_DIR='/path/to/datasets/'

# torchrun --nproc_per_node=4 eval_copy_detection.py --model carp --pretrained_weights $CARP400epMC --data_path $COPYDAYS_DATASET_DIR
# torchrun --nproc_per_node=4 eval_image_retrieval.py --imsize 512 --multiscale 1 --data_path $OXFORD_PARIS_DATASET_DIR --dataset roxford5k --model triplet --pretrained_weights ../../pretrained/triplet/release_ep940.pth
# torchrun --nproc_per_node=4 eval_image_retrieval.py --imsize 512 --multiscale 1 --data_path $OXFORD_PARIS_DATASET_DIR --dataset rparis6k --model carp --pretrained_weights $CARP200ep