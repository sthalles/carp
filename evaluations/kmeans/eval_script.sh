#!\bin\bash

DATASET_DIR='<path/to/datasets/>'

# python kmeans_eval_cifar10.py --model pclv2 --pretrained ../../pretrained/pcl_v2/PCL_v2_epoch200.pth.tar $DATASET_DIR
# python kmeans_eval_cifar100.py --model selav2 --pretrained ../../pretrained/sela_v2/selav2_400ep_2x224_pretrain.pth.tar $DATASET_DIR

# example runs
python kmeans_eval_dtd.py --model carp --pretrained ../../pretrained/carp/carp-200ep/checkpoint0199.pth $DATASET_DIR
python kmeans_eval_dtd.py --model carp --pretrained ../../pretrained/carp/carp-400ep/checkpoint0399.pth $DATASET_DIR
python kmeans_eval_dtd.py --model carp --pretrained ../../pretrained/carp/carp-400ep-lr=0.3/checkpoint0399.pth $DATASET_DIR
python kmeans_eval_dtd.py --model carp --pretrained ../../pretrained/carp/carp-200ep-mc/checkpoint0199.pth $DATASET_DIR
python kmeans_eval_dtd.py --model carp --pretrained ../../pretrained/carp/carp-400ep-mc/checkpoint0399.pth $DATASET_DIR
