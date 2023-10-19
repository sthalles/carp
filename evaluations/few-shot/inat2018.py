import torch.utils.data as data
from PIL import Image
import os
import json
import random
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


class INAT(data.Dataset):
    def __init__(self, root, ann_file, is_train=True, low_shot=False, k=1, transform=None):

        print("low_shot:", low_shot)

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print ('\t' + str(len(self.imgs)) + ' images')
        print ('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        self.transform = transform
        self.low_shot = low_shot

        if self.low_shot:
            self.convert_low_shot(k)

    def __getitem__(self, index):
        if self.low_shot:
            path = os.path.join(self.root, self.img_files_lowshot[index])
            species_id = self.labels_lowshot[index]
        else:
            path = os.path.join(self.root, self.imgs[index])
            species_id = self.classes[index]

        im_id = self.ids[index]
        img = self.loader(path)
        
        tax_ids = self.classes_taxonomic[species_id]

        if self.transform:
            img = self.transform(img)

        return img, im_id, species_id, tax_ids

    def convert_low_shot(self, k):

        label2img = {c:[] for c in range(8142)}

        for n in range(len(self.classes)):
            label2img[self.classes[n]].append(self.imgs[n])

        self.img_files_lowshot = []
        self.labels_lowshot = []

        for c,imlist in label2img.items():
            random.shuffle(imlist)
            self.labels_lowshot += [c]*len(imlist[:k])
            self.img_files_lowshot += imlist[:k]        

        assert len(self.img_files_lowshot) == len(self.labels_lowshot), f"{len(self.img_files_lowshot)} != {len(self.labels_lowshot)}"

    def __len__(self):
        if self.low_shot:
            return len(self.img_files_lowshot)
        else:
            return len(self.imgs)