
#!\bin\bash

CARP400ep='../../pretrained/carp/carp-400ep/checkpoint0399.pth'
CARP200epMC='../../pretrained/carp/carp-200ep-mc/checkpoint0199.pth'
CARP400epMC='../../pretrained/carp/carp-400ep-mc/checkpoint0399.pth'

# path to inat dataset
INAT_DATASET_DIR='/path/to/inat2018/'

# run carp on iNat-2018, vary the number of training examples k, remove --low-shot for full dataset evaluation
torchrun --nproc-per-node=4 evaluate_inat2018.py --k 1 --lr-scheduler multistep --model carp $INAT_DATASET_DIR $CARP200epMC --lr-classifier 0.3 --workers 8 --low-shot
torchrun --nproc-per-node=4 evaluate_inat2018.py --k 1 --lr-scheduler multistep --model carp $INAT_DATASET_DIR $CARP400epMC --lr-classifier 0.3 --workers 8 --low-shot

# path to voc07 dataset
VOC_DATASET_DIR='/path/to/voc/'

# run carp on VOC07, remove --low-shot for full dataset evaluation
torchrun --nproc-per-node=4 eval_svm_voc.py --pretrained $CARP400ep -a resnet50 --low-shot $VOC_DATASET_DIR
torchrun --nproc-per-node=4 eval_svm_voc.py --pretrained $CARP400epMC -a resnet50 --low-shot $VOC_DATASET_DIR

