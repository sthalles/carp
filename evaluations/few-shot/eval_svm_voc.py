from __future__ import print_function
import os
import sys
import torch
import argparse
import random
import numpy as np

from torchvision import transforms
import torchvision.models as models

from voc import Voc2007Classification
from sklearn.svm import LinearSVC

sys.path.append("../../")
from helpers.checkpoint_loader import CheckpointsLoader


def parse_option():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--cost', type=str, default='0.5')
    parser.add_argument('--seed', default=0, type=int)
    
    # model definition
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')    
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to pretrained checkpoint')


    # dataset
    parser.add_argument('--low-shot', default=False, action='store_true', help='whether to perform low-shot training.')    
    parser.add_argument('--model', default='carp', choices=['carp', 'selfcls', 'mocov3', 'pclv2', 'selav2', 'deepclusterv2',
                                                            'triplet', 'barlowtwins', "infomin", "dino", "swav"], type=str, help='model name')

    opt = parser.parse_args()

    opt.num_class = 20
    
    # if low shot experiment, do 5 random runs
    if opt.low_shot:
        opt.n_run = 5
    else:
        opt.n_run = 1
    return opt


def calculate_ap(rec, prec):
    """
    Computes the AP under the precision recall curve.
    """
    rec, prec = rec.reshape(rec.size, 1), prec.reshape(prec.size, 1)
    z, o = np.zeros((1, 1)), np.ones((1, 1))
    mrec, mpre = np.vstack((z, rec, o)), np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = 0
    for i in indices:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap

def get_precision_recall(targets, preds):
    """
    [P, R, score, ap] = get_precision_recall(targets, preds)
    Input    :
        targets  : number of occurrences of this class in the ith image
        preds    : score for this image
    Output   :
        P, R   : precision and recall
        score  : score which corresponds to the particular precision and recall
        ap     : average precision
    """
    # binarize targets
    targets = np.array(targets > 0, dtype=np.float32)
    tog = np.hstack((
        targets[:, np.newaxis].astype(np.float64),
        preds[:, np.newaxis].astype(np.float64)
    ))
    ind = np.argsort(preds)
    ind = ind[::-1]
    score = np.array([tog[i, 1] for i in ind])
    sortcounts = np.array([tog[i, 0] for i in ind])

    tp = sortcounts
    fp = sortcounts.copy()
    for i in range(sortcounts.shape[0]):
        if sortcounts[i] >= 1:
            fp[i] = 0.
        elif sortcounts[i] < 1:
            fp[i] = 1.
    P = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
    numinst = np.sum(targets)
    R = np.cumsum(tp) / numinst
    ap = calculate_ap(R, P)
    return P, R, score, ap


def main():
    args = parse_option()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
       
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)        
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),        
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = Voc2007Classification(args.data,set='trainval',transform = transform)
    val_dataset = Voc2007Classification(args.data,set='test',transform = transform)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=128)
    model.fc = torch.nn.Identity()

    file = open(os.path.join(os.path.dirname(args.pretrained), "voc_linear_eval.txt"), "a")
    file.writelines(f"------ Pascal VOC linear eval for {args.pretrained} ------\n")

    # load from pre-trained
    checkpoint_loader = CheckpointsLoader(args.pretrained)
    model = checkpoint_loader.load_pretrained(model, model_name=args.model)
            
    model.cuda()   
    model.eval()   

    test_feats = []
    test_labels = []    
    print('==> calculate test features')
    for idx, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        feat = model(images)
        feat = feat.detach().cpu()
        test_feats.append(feat)
        test_labels.append(target)

    test_feats = torch.cat(test_feats,0).numpy()
    test_labels = torch.cat(test_labels,0).numpy()
               
    test_feats_norm = np.linalg.norm(test_feats, axis=1)
    test_feats = test_feats / (test_feats_norm + 1e-5)[:, np.newaxis]
    
    result={}
    map_std_at_k=[]
    map_at_k=[]
    
    if args.low_shot:
        k_list = [1,2,4,8,16] #number of samples per-class for low-shot classifcation
    else:
        k_list = ['full']
        
    for k in k_list:
        cost_list = args.cost.split(',')
        result_k = np.zeros(len(cost_list))
        for i,cost in enumerate(cost_list):
            cost = float(cost)
            avg_map = []
            for run in range(args.n_run):
                if args.low_shot: # sample k-shot training data
                    # print('==> re-sampling training data')
                    train_dataset.convert_low_shot(k)    
                # print("Train data size:", len(train_dataset))

                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

                train_feats = []
                train_labels = []
                # print('==> calculate train features')
                for idx, (images, target) in enumerate(train_loader):
                    images = images.cuda(non_blocking=True)
                    feat = model(images)
                    feat = feat.detach()  

                    train_feats.append(feat)                
                    train_labels.append(target)

                train_feats = torch.cat(train_feats,0).cpu().numpy()
                train_labels = torch.cat(train_labels,0).cpu().numpy()

                train_feats_norm = np.linalg.norm(train_feats, axis=1)
                train_feats = train_feats / (train_feats_norm + 1e-5)[:, np.newaxis]

                # print('==> training SVM Classifier')
                cls_ap = np.zeros((args.num_class, 1))
                test_labels[test_labels==0] = -1 
                train_labels[train_labels==0] = -1 
                for cls in range(args.num_class):
                    clf = LinearSVC(
                        C=cost, class_weight={1: 2, -1: 1}, intercept_scaling=1.0,
                        penalty='l2', loss='squared_hinge', tol=1e-4,
                        dual=True, max_iter=2000, random_state=0)
                    clf.fit(train_feats, train_labels[:,cls])

                    prediction = clf.decision_function(test_feats)                                      
                    P, R, score, ap = get_precision_recall(test_labels[:,cls], prediction)
                    cls_ap[cls][0] = ap*100
                mean_ap = np.mean(cls_ap, axis=0)

                # print('==> Run%d mAP is %.2f: '%(run,mean_ap))
                avg_map.append(mean_ap)

            avg_map = np.asarray(avg_map)   
            log1 = 'Cost:%.2f - Average ap is: %.2f' %(cost,avg_map.mean())  
            log2 = 'Cost:%.2f - Std is: %.2f' %(cost,avg_map.std())
            
            # print(log1)
            # print(log2)

            file.writelines(log1 + "\n")
            file.writelines(log2 + "\n")

            result_k[i]=avg_map.mean()
            map_std_at_k.append(avg_map.std())
            map_at_k.append(avg_map.mean())
        result[k] = result_k.max()    
    # print(result)

    print("\t".join(str(v) for v in k_list))
    print("\t".join(str(v) for v in map_at_k))
    print("\t".join(str(v) for v in map_std_at_k))

    file.close()    
    
if __name__ == '__main__':
    main()

