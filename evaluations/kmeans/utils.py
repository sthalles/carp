import urllib.request
from torchvision import datasets
import os

class PrintMultiple(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

class ImageFolderWithIndices(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithIndices, self).__getitem__(index)
        # make a new tuple that includes original and the index
        tuple_with_path = (original_tuple + (index,))
        return tuple_with_path
    

def imagenet_subset_samples(dataset, traindir, label_subset):
    # extract subset of training images
    subset_file = urllib.request.urlopen(
        "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/" +
        str(label_subset) + "percent.txt")
    labeled_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]
    # update dataset
    dataset.samples = [(os.path.join(traindir, li.split('_')[0], li), dataset.class_to_idx[li.split('_')[0]])
                       for li in labeled_imgs]
    return dataset