import argparse
import os
import random
import time
import warnings
import utils
import sys
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from PIL import Image
from utils import AverageMeter, ProgressMeter

sys.path.append("../../")
from helpers.checkpoint_loader import CheckpointsLoader
from helpers.clustering import kmeans_classifier



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Kmeans Evaluation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--train-batch-size', default=256, type=int,
                    help='train set batch size')
parser.add_argument('-p', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--label-subset', default="10", type=str, choices=["1", "10", "100"],
                    help='percentage of labeled data: 1%, 10% or 100% (default: 1)')
parser.add_argument('--num-classes', default=100, type=int,
                    help='number of classes (1000 for ImageNet, 10 (default: 0.1)')
parser.add_argument('--load-features', action='store_true',
                    help='use features from earlier dump (in args.save_path)')
parser.add_argument('--backbone-dim', default=2048, type=int,
                    help='backbone dimension size (default: %(default)s)')
parser.add_argument('--model', default='', choices=['carp', "deepclusterv2", "selav2", "dino", "swav", "pclv2"], 
                                                          type=str, help='model name')


class GTSRBIndex(datasets.GTSRB):

    def __init__(self, *args, **kwargs):
        super(GTSRBIndex, self).__init__(*args, **kwargs)

    def __getitem__(self, index):

        path, target = self._samples[index]
        sample = Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


def main():
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # save log file
    sys.stdout = utils.PrintMultiple(sys.stdout, open(os.path.join(args.save_path, 'log.txt'), 'a+'))
    print(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    model.fc = nn.Identity()

    # load from pre-trained, before DistributedDataParallel constructor
    checkpoint_loader = CheckpointsLoader(args.pretrained)
    model = checkpoint_loader.load_pretrained(model, model_name=args.model)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset  = GTSRBIndex(args.data, split = 'train', transform = transform, download = True)
    val_dataset  = GTSRBIndex(args.data, split = 'test', transform = transform, download = True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    f = open(os.path.join(os.path.dirname(args.pretrained), "kmeans_eval_cifar10.txt"), 'a')
    f.writelines(f"------ K-means eval for {args.pretrained} ------\n")

    if args.load_features:
        train_features = np.load(os.path.join(args.save_path, "trainfeat.npy"))
        val_features = np.load(os.path.join(args.save_path, "valfeat.npy"))
        val_labels = np.load(os.path.join(args.save_path, "vallabels.npy"))
    else:
        train_features, _ = inference(train_loader, model, args, prefix='Train Set Inference: ')
        val_features, val_labels = inference(val_loader, model, args, prefix='Test Set Inference: ')

        # dump
        np.save(os.path.join(args.save_path, "trainfeat"), train_features)
        np.save(os.path.join(args.save_path, "valfeat"), val_features)
        np.save(os.path.join(args.save_path, "vallabels"), val_labels)

    # evaluate kmeans classifier
    print("Features are ready!\nEvaluate K-Means Classifier.")
    # kmeans_classifier(train_features, val_features, val_labels, args)
    val_nmi, val_adjusted_nmi, val_adjusted_rand_index, val_fms, val_homogeneity, val_completeness, val_v_measure = kmeans_classifier(train_features, val_features, val_labels, f, args)
    f.writelines(f'=> NMI: {val_nmi * 100.0}\n')
    f.writelines(f'=> Adjusted NMI: {val_adjusted_nmi * 100.0}\n')
    f.writelines(f'=> Adjusted Rand-Index: {val_adjusted_rand_index * 100.0}\n')
    f.close()

@torch.no_grad()
def inference(loader, model, args, prefix):
    all_features = np.zeros((len(loader.dataset), args.backbone_dim), dtype=np.float32)
    all_labels = np.zeros((len(loader.dataset), ), dtype=np.int32)
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix=prefix)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, targets, indices) in enumerate(loader):
        if torch.cuda.is_available():
            images = images.cuda()

        # compute output
        output = model(images)

        # compute prediction
        all_features[indices] = output.detach().cpu().numpy()
        # save labels
        all_labels[indices] = targets.numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return all_features, all_labels


if __name__ == '__main__':
    main()